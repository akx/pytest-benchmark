Calibration
===========

``pytest-benchmark`` will run your function multiple times between measurements. A `round` is that set of runs done between
measurements. This is quite similar to the builtin ``timeit`` module but it's more robust.

The problem with measuring single runs appears when you have very fast code. To illustrate:

.. image:: https://github.com/ionelmc/pytest-benchmark/raw/master/docs/measurement-issues.png
    :alt: Diagram illustrating issues with measuring very fast code

In other words, a `round` is a set of runs that are averaged together, those resulting numbers are then used to compute the
result tables. The default settings will try to keep the round small enough (so that you get to see variance), but not too
small, because then you have the timer calibration issues illustrated above (your test function is faster than or as fast
as the resolution of the timer).

By default ``pytest-benchmark`` will try to run your function as many times needed to fit a `10 x TIMER_RESOLUTION`
period. You can fine tune this with the ``--benchmark-min-time`` and ``--benchmark-calibration-precision`` options.

How many rounds?
----------------

By default the number of rounds is decided up front:
as many as fit in ``--benchmark-max-time``, but at least ``--benchmark-min-rounds``.

``--benchmark-precision`` makes the stopping point adaptive.
The relative margin of error of the mean is computed as the run goes on in batches of rounds,
and rounds stop once it falls below the target, so ``--benchmark-precision=0.02`` and ``--benchmark-confidence`` at
its default 99% means "stop once the true mean is within ±2% of the measured mean, with 99% confidence".
The usual bounds still apply: never less than ``--benchmark-min-rounds``, never longer than ``--benchmark-max-time``.

Because rounds aren't independent or normally distributed due to machine-level noise, doing a basic
``stddev/sqrt(rounds)`` doesn't give the correct precision. So, we batch rounds together to get better estimates,
and we use a Student's t test instead of a normal distribution because we have fewer degrees of freedom due to using
batches. Due to these properties, there are some results that you should be aware of:

* A run needs at least 21 rounds before it can stop at all, no matter how steady the timings are.
  21 is derived fromm ``PRECISION_MIN_BATCHES + PRECISION_CONFIRMATIONS - 1``.
* If ``--benchmark-max-time`` runs out before the target is reached, that benchmark emits a warning
  instead of silently reporting an unreproducible mean. Raise ``--benchmark-max-time``, loosen
  ``--benchmark-precision``, or accept that that particular benchmark is noisy.
