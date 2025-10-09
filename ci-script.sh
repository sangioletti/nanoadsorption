#!/bin/bash

# See: https://sipb.mit.edu/doc/safe-shell/
set -euf -o pipefail

# Get the directory of this script in a relatively robust fashion (pun intended). See:
# http://www.binaryphile.com/bash/2020/01/12/determining-the-location-of-your-script-in-bash.html
here=$(cd "$(dirname "$BASH_SOURCE")"; cd -P "$(dirname "$(readlink "$BASH_SOURCE" || echo .)")"; pwd)

cd $here/rust
cargo run --release
cp adsorption_rust.dat ..

cd $here
uv run main.py
