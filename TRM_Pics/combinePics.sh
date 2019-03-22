#!/bin/bash

mkdir Combined
mkdir Combined/{A..Y}
rm -r Combined/J

for i in {A..Y} ; do
  cp */"$i"/*.jpg Combined/"$i"/
done

mkdir Combined_no_Nikita
mkdir Combined_no_Nikita/{A..Y}
rm -r Combined_no_Nikita/J

for i in {A..Y} ; do
  cp M*/$i/*.jpg Combined_no_Nikita/"$i"/
  cp R*/"$i"/*.jpg Combined_no_Nikita/"$i"/
  cp T*/"$i"/*.jpg Combined_no_Nikita/"$i"/
done

mkdir Combined_no_Rosemond
mkdir Combined_no_Rosemond/{A..Y}
rm -r Combined_no_Rosemond/J

for i in {A..Y} ; do
  cp M*/$i/*.jpg Combined_no_Rosemond/"$i"/
  cp N*/"$i"/*.jpg Combined_no_Rosemond/"$i"/
  cp T*/"$i"/*.jpg Combined_no_Rosemond/"$i"/
done

mkdir Combined_no_Trung
mkdir Combined_no_Trung/{A..Y}
rm -r Combined_no_Trung/J

for i in {A..Y} ; do
  cp M*/$i/*.jpg Combined_no_Trung/"$i"/
  cp R*/"$i"/*.jpg Combined_no_Trung/"$i"/
  cp N*/"$i"/*.jpg Combined_no_Trung/"$i"/
done

mkdir Combined_no_Michael
mkdir Combined_no_Michael/{A..Y}
rm -r Combined_no_Michael/J

for i in {A..Y} ; do
  cp N*/$i/*.jpg Combined_no_Michael/"$i"/
  cp R*/"$i"/*.jpg Combined_no_Michael/"$i"/
  cp T*/"$i"/*.jpg Combined_no_Michael/"$i"/
done
