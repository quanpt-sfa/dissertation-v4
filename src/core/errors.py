"""Explicit failures raised while compiling the P0 protocol."""


class ConfigurationError(ValueError):
    """Base configuration error with actionable source context."""


class MissingModuleError(ConfigurationError):
    """A manifest-declared module is absent."""


class UndeclaredConfigurationFileError(ConfigurationError):
    """A YAML file is not listed by the sole manifest."""


class DuplicateOwnershipError(ConfigurationError):
    """A namespace or path has more than one owner."""


class UnknownReferenceError(ConfigurationError):
    """A configuration reference cannot be resolved."""


class ArtifactPathCollisionError(ConfigurationError):
    """Two declared artifacts could resolve to one path."""


class DecisionTraceabilityError(ConfigurationError):
    """D01--D45 traceability is incomplete or invalid."""


class MethodologicalInvariantError(ConfigurationError):
    """A locked methodological rule has been violated."""


class GeneratedFileDriftError(ConfigurationError):
    """A checked-in generated document differs from configuration output."""


class AccessPolicyError(ConfigurationError):
    """A declared step violates an access firewall."""
