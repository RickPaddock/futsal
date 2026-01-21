# PROV: FUSBAL.PIPELINE.MAIN.01
# REQ: FUSBAL-V1-OUT-001, SYS-ARCH-15
# WHY: Allow `python -m fusbal_pipeline ...` as a stable entrypoint for the pipeline CLI.

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
