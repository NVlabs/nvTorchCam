
#!/bin/bash

echo -e '\n>>> Testing cameras'
python tests/test_cameras.py

echo -e '\n>>> Testing warpings'
python tests/test_warpings.py

echo -e '\n>>> Testing utils'
python tests/test_utils.py