"""
Helper script to format Notion documentation exports.

Removes the Notion hash at the end of all files in the Documentation folder if
they have the file extension(s) defined in script settings below. Moves and
renames any README file(s) to the main section of the project, overwriting if
necessary.
"""

import os

# Script settings
fileExtensions = ['.md']                            # File extensions to remove the Notion hash from
readmeFileSubstrings = ['README',                   # Filename substrings to identify README files
                        'Documentation Homepage']   # Files containing these substrings will be renamed to README and moved to the main repo folder

docsDir = os.path.join(os.path.dirname(os.getcwd()), 'Documentation')
for entry in os.scandir(docsDir):
    for fileEx in fileExtensions:
        if entry.name.endswith(fileEx):
            # Remove the Notion hash
            newName = entry.name[:entry.name.rfind(' ')] + fileEx
            if any([substring.lower() in entry.name.lower() for substring in readmeFileSubstrings]):
                # File is a README, rename to README and move to the main repo folder
                newPath = os.path.join(os.path.dirname(docsDir), 'README' + fileEx)
            else:
                # File is not a README, just rename it in its location
                newPath = entry.path[:entry.path.rfind(' ')] + fileEx
            # Move/rename the file (overwrites if necessary)
            os.replace(entry.path, newPath)
