"""YAML configuration handling helper.

The implementation is taken from the previous ``clwm/utils/config_manager``
module and moved into a *top-level* file to avoid the now removed
``utils`` package.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import yaml

# -------------------------------------------------------------------------
# Private view class ------------------------------------------------------
# -------------------------------------------------------------------------


class _ConfigView:
    __slots__ = ("_data",)

    def __init__(self, data: Mapping[str, Any]):
        self._data = data

    # Mapping interface ------------------------------------------------
    def __getitem__(self, item: str):
        return self._wrap(self._data[item])

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    # Dict convenience helpers ----------------------------------------
    def get(self, key: str, default: Any = None):
        return self._wrap(self._data.get(key, default))

    # Attribute access -------------------------------------------------
    def __getattr__(self, name: str):
        try:
            return self._wrap(self._data[name])
        except KeyError as exc:
            raise AttributeError(name) from exc

    # Helpers ----------------------------------------------------------
    @staticmethod
    def _wrap(value: Any):
        if isinstance(value, Mapping):
            return _ConfigView(value)
        return value

    # Representation ---------------------------------------------------
    def __repr__(self) -> str:
        return f"ConfigView({self._data!r})"

    # -----------------------------------------------------------------
    # Public helpers ---------------------------------------------------
    # -----------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:  # noqa: D401 – explicit accessor
        """Return the **raw** underlying mapping.

        The training entry-point expects a *plain* ``dict`` so that it can be
        unpacked directly into the :class:`clwm.Trainer` constructor via the
        ``**`` syntax.  Exposing the internal representation avoids an extra
        ``yaml.safe_dump/load`` round-trip and keeps backwards-compatibility
        with earlier versions of the code base where :class:`Config` was just
        a thin ``dict`` wrapper.
        """

        # ``_data`` already contains regular Python containers – return a
        # shallow copy to prevent accidental mutations of the internal state.
        return dict(self._data)


# -------------------------------------------------------------------------
# Public Config class -----------------------------------------------------
# -------------------------------------------------------------------------


class Config(_ConfigView):
    """Thin wrapper around a dict that supports ``dot`` attribute access."""

    def __init__(self, mapping: Mapping[str, Any] | str | Path):
        if isinstance(mapping, (str, Path)):
            mapping = self._load_and_apply_defaults(mapping)
        super().__init__(mapping)

    # Convenience constructors ----------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        return cls(cls._load_and_apply_defaults(path))

    @classmethod
    def from_cli(
        cls, parser: argparse.ArgumentParser | None = None
    ) -> tuple["Config", argparse.Namespace]:
        if parser is None:
            parser = argparse.ArgumentParser(add_help=False)

        # Default command-line flags -----------------------------------
        parser.add_argument(
            "--config",
            type=str,
            default="config.yaml",
            help="Path to the YAML configuration file.",
        )

        parser.add_argument(
            "--categories",
            type=str,
            nargs="+",
            required=True,
            help="List of *category* names defined in atari.yaml to train on.",
        )

        parser.add_argument(
            "--zero-shot",
            action="store_true",
            help="When set, hold out one random game from the training "
            "categories for zero-shot evaluation.",
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=0,
            help="Global RNG seed (torch / numpy / python).",
        )

        parser.add_argument(
            "--size",
            type=str,
            default="5m",
            help=(
                "Model size preset to use - one of the keys defined under "
                "model.size in the YAML configuration (e.g. 1m, 5m, 20m)."
            ),
        )

        args = parser.parse_args()
        cfg = cls.from_yaml(args.config)

        # Expose CLI flags under a dedicated namespace -----------------
        cfg._data["cli"] = vars(args)  # type: ignore[attr-defined]

        # ------------------------------------------------------------------
        # Apply *size* preset ----------------------------------------------
        # ------------------------------------------------------------------
        # The YAML file contains a mapping ``model.size`` that holds
        # parameter presets keyed by an approximate *parameter count* label
        # ("1m", "5m", …).  When the user passes ``--size`` we merge the
        # corresponding dictionary into the *active* model configuration so
        # that downstream code can treat the values just like any other
        # manually specified ``dim / layers / heads`` trio.

        model_cfg: dict[str, Any] = cfg._data.setdefault("model", {})  # type: ignore[attr-defined]

        size_presets: dict[str, Any] = {
            k.lower(): v for k, v in model_cfg.get("size", {}).items()
        }

        # The YAML may define a *default* size label (``model.default``).
        default_label: str | None = model_cfg.get("default")

        # Determine the requested size label (CLI overrides YAML default).
        requested_label: str | None = (
            args.size.lower()
            if args.size
            else (
                default_label.lower()
                if isinstance(default_label, str)
                else None
            )
        )

        if requested_label is not None:
            if requested_label not in size_presets:
                avail = ", ".join(sorted(size_presets))
                raise ValueError(
                    f"Unknown model size '{requested_label}'. Available: {avail}"
                )

            # Merge (do not *replace*) so that explicit overrides – either
            # from the YAML file itself or via *future* dedicated CLI flags –
            # still take precedence.
            for k, v in size_presets[requested_label].items():
                model_cfg.setdefault(k, v)

            # Keep a record of the chosen label for logging/debugging.
            model_cfg.setdefault("size_label", requested_label)

        return cfg, args

    # Internal helpers -------------------------------------------------
    @staticmethod
    def _load_and_apply_defaults(path: str | Path) -> dict[str, Any]:
        with open(path, "r", encoding="utf-8") as fh:
            cfg: dict[str, Any] = yaml.safe_load(fh)

        cfg.setdefault("train", {})
        return cfg
