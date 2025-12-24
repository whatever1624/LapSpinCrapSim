"""Helper script to rename .md files in this folder to remove the Notion hash at the end of the filename"""

import os

cwd = os.getcwd()
for entry in os.scandir(cwd):
    if entry.name.endswith('.md'):
        newPath = entry.path[:entry.path.rfind(' ')] + ".md"
        os.rename(entry.path, newPath)
