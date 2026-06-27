# Lazy imports so that importing this package does NOT trigger
# heavy model-specific initializations (e.g. MatterGen opening its
# LMDB reference database) unless that suite is actually requested.

def __getattr__(name: str):
    if name == "MatterGenSuite":
        from models.suite.mattergen import MatterGenSuite
        return MatterGenSuite
    if name == "DiffCSPSuite":
        from models.suite.diffcsp import DiffCSPSuite
        return DiffCSPSuite
    if name == "SymmCDSuite":
        from models.suite.symmcd import SymmCDSuite
        return SymmCDSuite
    raise AttributeError(f"module 'models.suite' has no attribute {name!r}")
