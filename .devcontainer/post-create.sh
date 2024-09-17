#!/bin/bash
set -e

echo 'Starting post-create commands'
chmod -R 777 /workspace || echo 'chmod command failed'
pip install -e /workspaces/AModelADay/MyUtils || echo 'pip install command failed'
echo 'Post-create commands completed'