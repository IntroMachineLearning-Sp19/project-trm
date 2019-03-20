#!/bin/bash

mkdir Combined
mkdir Combined/{A..Y}
rm -r Combined/J

for i in {A..Y} ; do
  cp */"$i"/*.jpg Combined/"$i"/
done
