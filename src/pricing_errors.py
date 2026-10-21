"""User-correctable pricing input errors shared by pricing and risk services."""


class PricingInputError(ValueError):
    """A submitted option or market configuration is invalid."""
