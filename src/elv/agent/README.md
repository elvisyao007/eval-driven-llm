# agent/ — reserved extension point, intentionally not implemented

See DECISIONS.md ADR-0005. The agent / tool-calling surface is reserved in the
architecture so adding it later is not a rewrite. It is deliberately empty until
a concrete, real use case defines the requirements. Agent experimentation
happens in the content/radar layer, not here as a maintained product.
