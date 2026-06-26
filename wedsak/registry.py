from confit import Registry, set_default_registry, RegistryCollection


@set_default_registry
class WedsakRegistry(RegistryCollection):
    loss = Registry(("wedsak_registry", "loss"), entry_points=True)
    label_model = Registry(("wedsak_registry", "label_model"), entry_points=True)

    _catalogue = dict(
        loss=loss,
        label_model=label_model,
    )


registry = WedsakRegistry
