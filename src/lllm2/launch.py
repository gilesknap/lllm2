"""Curated launch selection; all tuning still comes from portable defaults."""

from . import config
from .defaults import starting_defaults
from .discovery import CATALOG, engines, hardware, models, probe
from .recommendations import PROFILES, fingerprint
from .settings import Settings, launch_args


def installed_models():
    found = models()
    for model in found:
        model["catalog_id"] = None
        model["identity_verified"] = False
        meta = model["metadata"]
        # Only fingerprint plausible measured checkpoints, never the whole model library.
        candidates = [
            p
            for p in PROFILES
            if p["model"]["size"] == model["size"]
            and not meta.get("error")
            and all(meta.get(k) == p["model"][k] for k in ("architecture", "mtp"))
        ]
        try:
            digest = fingerprint(model["path"]) if candidates else None
            profile = next(
                (p for p in candidates if p["model"]["sha256"] == digest), None
            )
            if profile:
                entry = next(
                    (
                        m
                        for m in CATALOG
                        if m.get("recommendation", {}).get("profile") == profile["id"]
                    ),
                    None,
                )
                if entry:
                    model.update(catalog_id=entry["id"], identity_verified=True)
        except (OSError, ValueError) as error:
            model["identity_error"] = str(error)
        if not model["catalog_id"]:
            # Unverified catalogue hints are useful in All models, but never acquire rank.
            candidates = [
                m
                for m in CATALOG
                if model["path"].endswith("/" + m["file"])
                and bool(m.get("mtp")) == bool(meta.get("mtp"))
            ]
            if len(candidates) == 1:
                model["catalog_id"] = candidates[0]["id"]
    ranks = {m["id"]: m.get("recommendation", {}).get("rank", 999) for m in CATALOG}
    return sorted(
        found,
        key=lambda m: (
            ranks.get(m["catalog_id"], 999) if m["identity_verified"] else 999,
            m["name"].casefold(),
            m["path"],
        ),
    )


def choose_launch(model_path="", engine_path="", backend="", device=""):
    """Choose a compatible tuple, or return a visible reason without changing preferences."""
    installed = installed_models()
    curated = {m["id"]: m["recommendation"] for m in CATALOG if m.get("recommendation")}
    if model_path:
        candidates = [m for m in installed if m["path"] == model_path]
        if not candidates:
            raise ValueError(
                "Selected checkpoint is no longer installed. Rescan or download the model."
            )
    else:
        candidates = [
            m
            for m in installed
            if m["identity_verified"] and m["catalog_id"] in curated
        ]
    if not candidates:
        return {
            "settings": None,
            "reason": "Download a recommended model, or choose an installed checkpoint in All models.",
        }
    builds = engines(refresh=False)
    if engine_path:
        builds = [e for e in builds if e["path"] == engine_path] or [probe(engine_path)]
    cards = hardware()["gpus"]
    if not cards:
        return {
            "settings": Settings(model=candidates[0]["path"]).dict(),
            "reason": "No NVIDIA GPU detected. Check GPU availability before starting.",
        }
    errors, first = [], None
    for model in candidates:
        profile_ids = (
            {curated[model["catalog_id"]]["profile"]}
            if model["catalog_id"] in curated
            else set()
        )
        profiles = [p for p in PROFILES if p["id"] in profile_ids]
        tuples = [
            (e, b, d)
            for e in builds
            for b in ([backend] if backend else ["CUDA", "Vulkan"])
            for d in e["devices"]
            if d.startswith(b) and (not device or d == device)
        ]
        tuples.sort(
            key=lambda t: (
                t[1] != "CUDA",
                not any(
                    p["backend"] == t[1] and p["engine"]["sha256"] == t[0].get("sha256")
                    for p in profiles
                ),
                t[0]["path"],
                t[2],
            )
        )
        for engine, selected_backend, selected_device in tuples:
            selection = Settings(
                model=model["path"],
                engine=engine["path"],
                backend=selected_backend,
                device=selected_device,
            )
            try:
                resolved = starting_defaults(selection)
                settings = Settings.parse(resolved["settings"])
                if first is None:
                    first = settings.dict()
                launch_args(settings, config.ENGINE_PORT)
                reason = (
                    ""
                    if not errors and model is candidates[0]
                    else "Using the first compatible installed recommendation. "
                    + " ".join(errors)
                )
                if not model_path and curated[model["catalog_id"]]["rank"] > min(
                    r["rank"] for r in curated.values()
                ):
                    reason = (
                        "The first recommendation is not available with a compatible verified checkpoint. Using the next compatible installed recommendation. "
                        + " ".join(errors)
                    )
                return resolved | {"reason": reason}
            except (OSError, ValueError, KeyError) as error:
                errors.append(f"{model['name']}: {error}")
    return {
        "settings": first
        or Settings(
            model=candidates[0]["path"],
            engine=engine_path,
            backend=backend or "CUDA",
            device=device or "",
        ).dict(),
        "reason": "No compatible engine configuration. Choose an installed GPU build under Customize settings. "
        + " ".join(errors[:2]),
    }
