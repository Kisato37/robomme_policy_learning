# Simulator import environment: startup repair

The first authorized end-to-end smoke at revision `a47a26f` stopped on its first
row before constructing an environment. No reset, robot action or scientific
outcome was produced. The separately completed architecture smoke did not
exercise simulator imports; its PASS must not be presented as end-to-end PASS.

## Cause and limited correction

The original dependency imports make three changes covered by the strict
environment check:

- OpenCV prepends its binary-library directory to `LD_LIBRARY_PATH`.
- SAPIEN sets `SAPIEN_PACKAGE_PATH` to its own package directory.
- The original RoboMME registry sets `TF_CPP_MIN_LOG_LEVEL=3`.

Previously the bootstrap compared the post-import process against the
pre-import environment. The corrected bootstrap verifies two separately
recorded phases without setting, restoring, or ignoring any live variable.

The simulator profile in a **new** hashed runtime-environment manifest may
declare `post_import_process_environment`, containing exactly the three keys
above and their exact reviewed post-import string values. The OpenCV prefix
must be one absolute directory followed by the entire unchanged startup
library path. Keep its literal spelling, including any `..` segments emitted by
the installed loader. This field does not change the process launch environment.

Before constructing a scene and before connecting the policy client, every
declared value must match the applicable phase. The bootstrap also records the
validated post-import environment in episode provenance. Unexpected sensitive
variables, changed CUDA/Vulkan mappings, and malformed transition declarations
remain hard errors. Policy processes cannot declare this simulator transition.
An absent transition never causes live changes to be accepted automatically.

## Preparing reviewed evidence and rerunning

1. In a fresh process with the exact simulator startup profile, import only the
   original wrappers/registry. Do not create a scene, reset an episode, load a
   policy or execute actions. Record all environment differences and hash the
   library source/config files that explain them.
2. Review those observations, then put the three expected values and source
   references in a new runtime manifest. Do not derive accepted values from
   whatever happens to be live inside a scientific worker.
3. Preserve the aborted run and all previous architecture evidence. Commit and
   synchronize only under the corresponding authorization. Re-run CPU checks
   on the target server. Bind new evidence to the new source/environment hashes.
4. Re-run architecture validation for the new binding, then the unchanged
   48-row development smoke under appropriate execution authority. Do not edit
   or rebind an old PASS, overwrite an attempt, silently resume mid-episode, or
   bypass the architecture/authorization gates.

This is a bootstrap/provenance correction, not a change to UK48/UN48, U32
preservation, capacity, memory features, model weights, tasks, seeds, numerical
settings, termination rules or statistical comparisons. Formal evaluation and
additional U trajectories remain separately gated.
