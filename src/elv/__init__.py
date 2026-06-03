"""elv — eval-driven reliable LLM systems.

The package is organized so the eval core (elv.eval) is independent of any
specific retriever, store, or model. Payload modules depend on eval, never the
reverse.
"""
__version__ = "0.0.1"
