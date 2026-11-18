"""Tiny image viewer, run as its own process: ``python show_image.py <png> [title]``.

The detection step launches it so the picture pops up without blocking the graph (GaP's
nodes run in worker threads, and this OpenCV build has no GUI). Any key or closing the
window ends it; the next detection replaces it.
"""

from __future__ import annotations

import sys
import tkinter as tk


def main() -> None:
    path = sys.argv[1]
    title = sys.argv[2] if len(sys.argv) > 2 else "rotor detection"
    root = tk.Tk()
    root.title(title)
    image = tk.PhotoImage(file=path)
    tk.Label(root, image=image).pack()
    root.bind("<Key>", lambda _e: root.destroy())
    root.attributes("-topmost", True)  # come to the front once, then behave normally
    root.after(500, lambda: root.attributes("-topmost", False))
    root.mainloop()


if __name__ == "__main__":
    main()
