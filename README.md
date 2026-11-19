# ZTF Supernova Stamp Classifier

A convolutional neural network, written in PyTorch, scores every public alert
from the Zwicky Transient Facility as it arrives and decides whether it looks
like a supernova. No ground truth exists at that point. Weeks later,
spectroscopy published by the Transient Name Server says what those objects
actually were, those labels are joined back onto predictions made long before,
and the model is scored against the ALeRCE broker's production classifier on
identical images.

It runs unattended on a home server: 3,675,399 alerts over 83 nights so far,
from a single cron entry.

<!-- TODO: uncomment once `bash publish_dashboard.sh --setup` has been run
**[Live dashboard](https://peckis.github.io/ztf-dashboard/)**, rebuilt nightly.
-->

### Result

233 spectroscopically confirmed supernovae against 10,567 stream controls, both
models scored on identical objects, 95% intervals from a paired bootstrap.

| | this work | ALeRCE | difference |
|---|---|---|---|
| AUC | 0.974 | 0.968 | +0.006 [−0.008, +0.022] |
| signal efficiency @ 10× background rejection | 0.910 | 0.923 | −0.013 [−0.059, +0.032] |
| signal efficiency @ 100× rejection | 0.738 | 0.794 | −0.056 [−0.124, +0.015] |
| signal efficiency @ 1000× rejection | 0.592 | 0.455 | +0.137 [−0.088, +0.315] |

No interval excludes zero, in either direction. The two classifiers are
statistically indistinguishable. An earlier measurement on 97 supernovae gave
+0.028 [+0.001, +0.063], which did not survive 2.4× more data.

### Beyond the model

- Predictions are made before the truth exists. Alerts are scored on arrival;
  TNS publishes what they were over the following weeks, and those labels are
  joined back onto the stored predictions.
- The loop recovers its own gaps after an outage, replays label feeds before
  they expire, and reports its own health with an exit code.
- Six results did not survive checking: circular labels, an epoch advantage, a
  0.917 floor from five numbers and no pixels, a reweighting that reversed a
  conclusion, an AUC resting on a single negative, and the +0.028 above.
- The claim is parity, not victory. No regime in this measurement shows this
  network to be better than the production baseline.

---

## Background

### The alert stream

The Zwicky Transient Facility [1] is a wide-field optical survey on the 48-inch
Samuel Oschin telescope at Palomar, with a 47 deg² camera. It images the
northern sky repeatedly. Each new exposure is aligned against a deep reference
template and subtracted from it, so that everything unchanged cancels and only
what varied survives. Every significant residual becomes an *alert*: three
63×63 pixel cutouts (new, reference, difference) plus about a hundred metadata
fields, serialised as Avro and distributed over Kafka within minutes [2].

ZTF emits of order 10⁵ alerts a night. Almost none are new astrophysical
events. The bulk are subtraction artefacts, known variable stars, active
galactic nuclei, and moving solar system objects. Supernovae are roughly 0.2%
of the stream, which makes this a rare-signal problem: a small interesting
population buried in backgrounds sharing its observable signature, a point-like
brightening where nothing was before. Rubin Observatory's LSST is designed for
about ten million alerts a night, so the scale is about to get worse.

Because the raw stream is too large to consume directly, *alert brokers* ingest
it, annotate each alert and redistribute a classified version. ALeRCE [3] is one
of the established ones. It runs two classifiers: a light-curve classifier once
an object has at least six detections, and a *stamp classifier* [4] applied to
the cutouts of an object's first alert, sorting it into supernova, AGN, variable
star, asteroid or bogus. The stamp classifier reports about 94% accuracy on a
balanced test set and builds rotational invariance into its architecture. It is
the baseline used throughout, and it is a model in production serving a real
community, not a reference implementation.

### Why images, and why the truth is late

On a first detection there is no light curve to classify. There is one
photometric point. Everything else is in the pixels: whether a host galaxy is
present, whether the source sits offset from a galactic nucleus or directly on
it, whether the residual looks like a real source or a bad subtraction.

Speed matters because supernovae evolve over weeks. A spectrum near peak is
scientifically valuable and the same spectrum two months later is not, so the
operational question is which of tonight's alerts deserve tonight's telescope
time. Telescope time being far scarcer than alerts, the useful working point is
high purity rather than high efficiency. A classifier that recovers every
supernova while flagging a quarter of the stream is no use to anyone.

Ground truth comes from the Transient Name Server, the IAU-sanctioned registry.
An object reported there gets a provisional designation, and if someone later
takes a spectrum the entry is updated with a type. That update arrives weeks to
months after the alert, because the spectrum has to be scheduled, taken, reduced
and reported. The delay is not an inconvenience to be engineered around. It is
the structural feature of the problem, and the reason this is built as a loop
rather than a script.

## Methods

### Data and model

Nightly public archives arrive as gzipped tar files of Avro packets, 0.6 to
16 GB each. They are streamed and parsed in place rather than unpacked.

Two preprocessing choices follow from the data rather than convention. First,
`arcsinh` rather than log compression: the difference channel is roughly half
negative by construction, since it is a subtraction and noise scatters either
side of zero, and `arcsinh` is logarithmic in the tails while staying linear
through zero. Second, normalisation statistics are computed on the training
split alone and then stored inside the checkpoint next to the weights.
Computing them over everything would leak test information, and applying
different statistics at inference than at training gives probabilities that look
plausible and mean nothing. The loader refuses a bare `state_dict` for that
reason. FITS sentinels become NaN, big-endian arrays become native float32, and
NaN maps to zero only after standardisation so that missing pixels land on the
channel mean instead of dragging it.

The network is small, 167,618 parameters, written in PyTorch:

```
3 × [Conv2d(3×3, pad 1) → BatchNorm2d → ReLU → MaxPool(2,2)]   3 → 32 → 64 → 128
AdaptiveAvgPool2d(3×3)                                         → [B, 128, 3, 3]
Flatten → Dropout(0.3) → Linear(1152, 64) → ReLU → Dropout(0.3) → Linear(64, 2)
```

Adam at `lr=1e-3` with cosine annealing over 40 epochs, batch size 64,
cross-entropy weighted inversely to class frequency. The checkpoint kept is the
best epoch by held-out AUC rather than the last, since test AUC plateaus and
then wobbles.

Augmentation is random flips and 90° rotations, the dihedral group of the
square, for an eightfold effective increase in data. The justification is
physical rather than generic: a stamp has no preferred orientation, and rotating
the sky does not change what a supernova is. The baseline exploits the same
symmetry, building it into the architecture instead of the training data [4].

### The nightly pipeline

`ingest_night.py` streams a night's archive, scores every alert in batches of
256, and writes to SQLite. Two measured constraints shape it: a night holds
55,000 to 340,000 alerts, and a stored stamp is 47 KB, so keeping all of them
would be about 1 TB a year. Every prediction is therefore stored at roughly 200
bytes a row, while stamps are kept only for the thousand highest-scoring alerts
per night, selected with a bounded min-heap so memory stays flat. That retained
the stamps of 187 of the 238 objects TNS later classified; the other 21% fell
below their night's cut and are unavailable for retraining, which is the price
of the storage bound.

Alerts from objects with more than two prior detections are skipped. The model
was trained on first detections, and repeat alerts for long-known objects are
out of distribution: an early version gave 0.9988 to the 367th detection of a
known AGN. This also matches the baseline's domain.

Labels come from TNS, joined on the field listing the survey designations an
object was reported under. A type beginning `SN` or `SLSN` is a positive; an
explicit allowlist of others (`AGN`, `CV`, `TDE`, `Nova`, `Star`,
`Impostor-SN`, and so on) is a negative; anything else, including a
reported-but-unclassified transient, stays unlabelled rather than guessed at.
TNS keeps daily delta files for two weeks, so the nightly job replays the last
seven days every run and recovers longer outages from the full catalogue.

Controls are not curated by class. They are ordinary stream alerts appearing
nowhere in TNS, taken deterministically at 300 per night so the evaluation set
is stable across rebuilds. A nightly crossmatch against SIMBAD, Milliquas and
the AAVSO Variable Star Index upgrades some from "presumed negative on the base
rate" to a positive identification as a star or AGN. A catalogued *galaxy* match
is never accepted, since supernovae occur in galaxies and such a match is
exactly what a real one looks like.

The whole loop runs from one cron entry. Gaps are found by asking the database
which nights it holds rather than assuming the previous run worked, so an outage
self-heals. A separate health check compares what should have happened against
what did and exits with a status code; it deliberately does not read logs, since
logs only record runs that happened, and the likeliest failures (scheduler
removed, disk full, stale lock) produce no log line at all.

Training runs on a laptop GPU, inference CPU-only on the server at about 1,400
alerts/second on four threads. That is not the bottleneck. A night takes roughly
23 minutes end to end, split between downloading a multi-gigabyte archive and
parsing some 130,000 Avro packets.

### Evaluation

The baseline is queried per object through ALeRCE's public API at a sustained
2 requests/second. Its `p_sn` is the probability it assigns to the supernova
class, the same quantity this network outputs, on the same images.

Three choices keep the comparison honest:

1. Both thresholds are tuned on training data and applied to test. Tuning one
   side only manufactures a win, and the baseline's optimum sits near 0.12, not
   0.5.
2. All comparisons use the objects both models scored. The baseline covers 43%
   of controls, limited by the API rate, and comparing each model on whatever it
   happened to cover would measure different populations.
3. Intervals come from a paired bootstrap of 2,000 resamples, scoring both
   models on the identical resampled set each draw so their correlated errors
   cancel and the interval applies to the difference.

## Results

### Dataset

| quantity | value |
|---|---|
| alerts scored | 3,675,399 |
| distinct objects | 3,616,529 |
| nights ingested | 83 |
| objects with a TNS spectroscopic type | 238 |
| — confirmed supernovae | 233 |
| — confirmed non-supernovae | 5 |
| in TNS, reported but not yet typed | 658 |
| stamps retained for retraining | 187 |

The evaluation set is 233 confirmed supernovae against 24,630 controls, of which
69 are positively identified as stars or AGN by catalogue crossmatch and the
rest are presumed negative on the base rate. The baseline has a score for 10,567
of them, and every head-to-head number below uses that shared subset.

### Discrimination

| model | AUC | 95% CI |
|---|---|---|
| this work | 0.9737 | [0.9647, 0.9818] |
| ALeRCE `stamp_classifier` | 0.9680 | [0.9524, 0.9809] |
| difference | +0.0059 | [−0.0076, +0.0220] |

The interval spans zero.

### Working points

AUC integrates over the whole ROC curve, including regions nobody operates in.
The operational question is signal efficiency at a fixed background rejection,
each model given its own threshold so both admit the same fraction of
background.

| background efficiency | rejection | this work | ALeRCE | difference |
|---|---|---|---|---|
| 10% | 10× | 0.910 | 0.923 | −0.013 [−0.059, +0.032] |
| 1% | 100× | 0.738 | 0.794 | −0.056 [−0.124, +0.015] |
| 0.1% | 1000× | 0.592 | 0.455 | +0.137 [−0.088, +0.315] |

No interval excludes zero at any working point. Note that AUC favours this
network while the 100× point favours the baseline; that is not a contradiction,
since a model can rank better on average while being worse in the narrow region
where the decision is made. It is the reason both are reported.

The 1000× row has the largest effect and the least support. With 10,567
controls, a 0.1% background efficiency puts the threshold at roughly the tenth
highest control score, so a handful of objects determine it. The width of that
interval is the honest statement, not the +0.137.

### Calibration

The two models' raw scores are not comparable, so no fixed-threshold comparison
between them is meaningful.

| at threshold 0.5 | signal efficiency | background efficiency |
|---|---|---|
| this work | 0.979 | 0.238 |
| ALeRCE | 0.356 | 0.0008 |

This network reaches 97.9% signal efficiency at 0.5 while admitting 23.8% of
background: good ranking, poor calibration, a consequence of class weighting on
a set roughly 4:1 in favour of positives against a stream that is 0.2%
supernovae. The baseline has the opposite problem, structurally rather than as a
defect. As one head of a five-class softmax its `p_sn` carries no mass above 0.9
at all, and its 1000× rejection point sits at 0.419 where this network's sits at
0.997.

## Discussion

### What a tie means

The goal was never to beat ALeRCE. ALeRCE is a production broker serving the
time-domain community, and its stamp classifier is one component of a system
that also runs a light-curve classifier, outlier detection, crossmatch services
and a public API. This project reimplements one head of one of those models.

An independently built, independently labelled classifier reaches parity with
it. That is the claim, and it is deliberately modest, since no regime in this
measurement shows this network to be better.

It is worth saying what makes parity less impressive than it sounds. I trained
on about 14,000 confirmed positives, narrowed the task from five classes to two,
and used a more recent architecture, and still only matched. That says more
about the quality of the baseline than about this work.

### Relation to rare-signal searches

The statistical problem is the familiar one from searches for rare processes at
colliders: a signal of order 0.2% of the sample, backgrounds sharing the
observable signature, and a decision taken at high purity rather than high
efficiency. The quantities reported above are the usual ones, and the lessons
that transfer are not about supernovae. A classifier trained on labels derived
from the model it is benchmarked against measures imitation rather than
accuracy. A control sample assembled differently from the signal sample will
differ in ways the network can exploit, and a nuisance-only baseline is the
cheapest way to find out. A difference between two discriminants needs a paired
interval, because they are evaluated on the same events and their errors are
correlated.

### Limitations

**Selection on the positives.** Both models are scored only on supernovae
somebody judged worth a spectrum. Neither can be measured on the ones the
community missed, and this biases both in the same unknown direction.

**Possible circularity favouring the baseline.** ALeRCE reports its own stamp
classifier's candidates to TNS, and 971 of those were spectroscopically
confirmed between 2019 and 2021 [4]. Any confirmed supernova reaching TNS by
that route is, by construction, one the baseline scored highly, so the
evaluation positives may be mildly enriched in its own successes. TNS receives
reports from many independent surveys, so the effect is diluted, and it is not
quantified here. The direction matters though: it biases the comparison against
this work rather than toward it.

**Presumed negatives.** Only 69 of 24,630 controls carry a positive catalogue
identification. The rest rest on the base-rate argument.

**Only new objects.** Alerts from objects with more than two prior detections
are not scored, so neither number here describes performance on well-sampled
objects.

**Subtype coverage.** Of the 233 evaluation supernovae, 71% are Type Ia, 22%
Type II, 6% Type Ib/Ic and 1% superluminous. The training sample has the same
composition to within a percent, so there is no shift between them, but both
reflect what follow-up spectroscopy targets rather than the true population. The
rare classes are too sparse to measure: one Type Ibn, no Type Icn.

**Random rather than temporal split.** A split by date (train ≤2023, test ≥2024)
would be stricter and is not yet done.

**Baseline coverage.** Scores were retrieved for 10,567 of 24,630 controls,
limited by the API rate. The comparison is restricted to that subset rather than
extrapolated across it.

## Troubles I ran into

Six times a number looked great and turned out to be a consequence of how I'd
set the measurement up rather than anything about the model. They're listed
because the final result only means something if you know what got thrown out on
the way to it.

**The first benchmark was circular.** I built the training set from ALeRCE's own
classifications, trained on it, and then benchmarked against ALeRCE. AUC 0.9944.
I was very pleased with myself for about a day, until I worked out that I hadn't
built a classifier so much as an expensive way of agreeing with one. Everything
after this point uses TNS spectroscopy for labels.

**I was showing my model the good photo.** The stamps I'd fetched for confirmed
supernovae came from whatever epoch the API handed back, and roughly a quarter
were near peak brightness. ALeRCE scores the first detection, which is fainter
and messier. Pinning both sides to the first detection killed a "40% fewer false
alarms" claim I had already written down. (Lesson learned about writing things
down.)

**Most of this isn't about what the thing looks like.** I took five numbers off
each stamp (background level, peak brightness, difference signal-to-noise, noise
depth, and whether something was already sitting there in the template), threw
away every pixel, and handed them to a gradient-boosted tree. AUC 0.917. The CNN
manages 0.974, so it does learn something from the shape of the source, but much
less than the raw number suggests. Any image classifier here has to be read
against that floor.

**Reweighting flipped the conclusion.** I'd deliberately packed the training set
with hard negatives, long-lived variable objects that look transient. They ended
up as 59% of my test negatives, where a real night is about 16%. Reweighting to
the actual stream mix turned an AUC win into a false positive rate of 9.5%
against ALeRCE's 3.8%. Identical predictions, opposite conclusion, which I found
genuinely unsettling.

**The dashboard once declared victory over one object.** It reported I was
"genuinely ahead" on the basis of 24 confirmed supernovae and exactly one
confirmed non-supernova. With a bootstrap confidence interval, no less. What it
had actually computed was the fraction of supernovae outscoring a single
asteroid. (The code now refuses to quote an interval below 20 objects and 5 of
either class.)

**The headline result didn't survive more data.** On 97 confirmed supernovae I
measured +0.028 AUC over ALeRCE, 95% interval [+0.001, +0.063]. Significant,
just about. I backfilled 19 more nights, reached 233 supernovae, and it
collapsed to +0.006 [−0.008, +0.022]. The old estimate sits outside the new
interval, so it wasn't merely imprecise, it was wrong. I'd added the paired
bootstrap to the nightly export the evening before, and it printed SPANS ZERO on
its own.

## References

1. Bellm, E. C. et al. 2019, *The Zwicky Transient Facility: System Overview,
   Performance, and First Results*, PASP **131**, 018002.
   [arXiv:1902.01932](https://arxiv.org/abs/1902.01932)
2. Patterson, M. T. et al. 2019, *The Zwicky Transient Facility Alert
   Distribution System*, PASP **131**, 018001.
3. Förster, F. et al. 2021, *The Automatic Learning for the Rapid
   Classification of Events (ALeRCE) Alert Broker*, AJ.
   [arXiv:2008.03303](https://arxiv.org/abs/2008.03303)
4. Carrasco-Davis, R. et al. 2021, *Alert Classification for the ALeRCE Broker
   System: The Real-time Stamp Classifier*, AJ.
   [arXiv:2008.03309](https://arxiv.org/abs/2008.03309)
