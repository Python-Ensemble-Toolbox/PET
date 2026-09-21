"""Binding an analysis to a scheme rather than inheriting one.

Lets a scheme take its analysis flavour as an argument, so one class covers
``approx``/``full``/``subspace`` instead of one class per combination.

How a scheme ends up paired with an analysis
--------------------------------------------
Every scheme declares, right on the class,
which flavours it supports and which class handles each -- e.g.
``esmda.py``::

    class ESMDA(AnalysisBindingMixin, ...):
        COMPATIBLE_ANALYSES = {
            "approx": approx_update,
            "full": full_update,
            "subspace": subspace_update,
        }

Walked through for ``ESMDA(da, en, sim, analysis="approx")``, at construction
time::

    1. ESMDA.__init__(...)            [pipt/update_schemes/esmda.py]
           |
           |  self.bind_analysis(self.resolve_analysis(analysis, keys_da))
           v
    2. resolve_analysis("approx", keys_da) -> "approx"     [this module]
           picks the flavour: explicit argument, else keys_da["analysis"],
           else "approx".
           |
           v
    3. bind_analysis("approx")                              [this module]
           looks "approx" up in `self.COMPATIBLE_ANALYSES`, giving
           approx_update. self.analysis = approx_update(self) -- an
           *instance*, holding a reference back to the scheme (`self`) it
           was built from.

    Later, once per iteration:

    4. ESMDA.calc_analysis() calls self.update(enX=..., enY=..., ...)
           |
           |  AnalysisBindingMixin.update() just forwards:
           v
       self.analysis.update(enX=..., enY=..., ...)           [analysis/approx.py]
           does the linear algebra, reading whatever context it needs off
           `self.scheme` -- the esmda_instance from step 3. `scheme.lam` is
           the scheme's own attribute; `scheme.keys_da` is its ensemble's,
           exposed as a property on the scheme (see AssimilationScheme).
           The analysis does not need to know which is which.

``EnKF``/``ES`` never revisit a data group, so the prior-increment term
``full`` adds over ``approx`` never applies -- the two produce identical
output (pinned by the characterisation suite). Rather than special-casing
that in code, ``EnKF.COMPATIBLE_ANALYSES`` just points ``"full"`` at the same
class as ``"approx"``:

    COMPATIBLE_ANALYSES = {"approx": approx_update, "full": approx_update, "subspace": subspace_update}

``ES`` inherits this dict unchanged, so the fact lives in exactly one place
and applies regardless of how the scheme was constructed.

Mixing in is still supported, but nothing live uses it
--------------------------------------------------------------------------
``bind_analysis`` still checks whether an analysis was mixed directly into
the scheme's bases (``_flavour_is_mixed_in``) and, if so, leaves
``self.analysis`` unset and lets that inherited ``update()`` take over
instead of building one. Both flavours that used to need this --
``hybrid_update`` (multilevel ES-MDA) and ``margIS_update`` (marg-IS) -- now
bind normally instead: both take the same ``(enX, enY, enE, **kwargs)``
shape as ``approx_update`` and friends, so ``esmda_hybrid.COMPATIBLE_ANALYSES
= {"hybrid": hybrid_update}`` and ``GNEnRML.COMPATIBLE_ANALYSES["margis"] =
margIS_update`` bind them the normal way.

Mixing an analysis directly into a scheme's bases is riskier than it looks
when the scheme base is listed first, which it usually must be: whichever
class the scheme's own ``__init__`` needs to resolve to has to come first,
but that can leave the *analysis's* ``update()`` shadowed by
``AnalysisBindingMixin.update()`` -- found first via the scheme's own MRO chain --
regardless of what ``bind_analysis`` decides. That bit both ``esmda_hybrid``
and ``gnenrml_margis`` (the latter fixed with an explicit ``update``
override before margis was converted to bind normally; see the CHANGELOG).
No class in this repository mixes an analysis in any more; ``co_lm_enrml``,
the last one, is a thin ``LMEnRML`` subclass that binds normally.
Prefer binding (a ``COMPATIBLE_ANALYSES`` entry) over mixing in for any new
flavour that fits the ``(enX, enY, enE, **kwargs)`` shape; mixing in is only
for an analysis that genuinely cannot, the way ``margIS_update`` used to.
"""

__all__ = ["AnalysisBindingMixin"]


class AnalysisBindingMixin:
    """Resolve an analysis flavour to an analysis object and delegate to it."""

    #: Flavour name -> analysis class to build with ``self`` as its scheme.
    #: Every scheme mixing this in sets its own (see module docstring). A
    #: scheme that instead gets a flavour by mixing the analysis directly
    #: into its bases needs no entry for it here, since ``bind_analysis``
    #: never consults this dict in that case.
    COMPATIBLE_ANALYSES: dict[str, type] = {}

    #: The bound analysis object, or ``None`` when a mixin supplies the
#: flavour instead. ``analysis_name`` holds the flavour's name.
    analysis = None

    def resolve_analysis(self, analysis=None, keys_da=None) -> str:
        """Decide the flavour: explicit argument, else the config, else "approx"."""
        if analysis is not None:
            return str(analysis).lower()
        if keys_da is not None:
            return str(keys_da.get("analysis", "approx")).lower()
        return "approx"

    def bind_analysis(self, analysis) -> None:
        """Bind the analysis for ``analysis``, unless a mixin already supplies one.

        Nothing shipped in this repository takes that path today (see the
        module docstring); it remains for a scheme that mixes an analysis
        directly into its bases instead of listing it in
        ``COMPATIBLE_ANALYSES``, in which case it keeps the inherited
        implementation and binds nothing.
        """
        self.analysis_name = analysis
        if self._flavour_is_mixed_in():
            self.analysis = None
            return
        if analysis not in self.COMPATIBLE_ANALYSES:
            raise KeyError(
                f"{type(self).__name__} has no {analysis!r} analysis flavour. "
                f"Available: {', '.join(sorted(self.COMPATIBLE_ANALYSES))}."
            )
        self.analysis = self.COMPATIBLE_ANALYSES[analysis](self)

    def _flavour_is_mixed_in(self) -> bool:
        """True if some other class in the MRO already defines ``update``."""
        return any(
            "update" in klass.__dict__
            for klass in type(self).__mro__
            if klass is not AnalysisBindingMixin
        )

    def update(self, *args, **kwargs):
        """Delegate the analysis step to the bound analysis.

        Only reached when nothing else in the MRO defines ``update``; a
        mixed-in flavour takes precedence and never gets here.
        """
        if self.analysis is None:
            raise AttributeError(
                f"{type(self).__name__} has no analysis bound and no "
                f"mixed-in update(); bind_analysis() was not called."
            )
        return self.analysis.update(*args, **kwargs)
