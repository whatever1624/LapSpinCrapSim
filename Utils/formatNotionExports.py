"""
Helper script to format Notion documentation exports.

Removes the Notion hash at the end of all files in the Documentation folder if
they have the file extension(s) defined in script settings below. Moves the
README file(s) to the main section of the project, overwriting if necessary.
"""

import os

# Script settings
fileExtensions = ['.md']    # File extensions to remove the Notion hash from

docsDir = os.path.join(os.path.dirname(os.getcwd()), 'Documentation')
for entry in os.scandir(docsDir):
    for fileEx in fileExtensions:
        if entry.name.endswith(fileEx):
            # Remove the Notion hash
            newName = entry.name[:entry.name.rfind(' ')] + fileEx
            if 'README' in entry.name.upper():
                # File is a README, move it to the main repo
                newPath = os.path.join(os.path.dirname(docsDir), newName)
            else:
                # File is not a README, just rename it in its location
                newPath = os.path.join(docsDir, newName)
            # Move/rename the file (overwrites if necessary)
            os.replace(entry.path, newPath)
