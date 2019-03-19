#!/bin/bash

mkdir Combined
mkdir Combined/{A..Y}
rm Combined/J

for i in {A..Y} ; do
  cp */"$i"/* Combined/"$i"/
done
