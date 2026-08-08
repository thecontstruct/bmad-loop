"""``python -m bmad_loop`` entry point — delegates to the console-script main().

Mirrors the ``if __name__ == "__main__"`` guard at the foot of ``cli.py`` so the
module form and the installed ``bmad-loop`` script share one dispatch path.
"""

import sys

from bmad_loop.cli import main

if __name__ == "__main__":
    sys.exit(main())
