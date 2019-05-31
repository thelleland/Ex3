
# safety first
set -o pipefail
shopt -s failglob 

# resize to 299x299, but keep image size
# drop -resize if zooming with crop is better (but maybe not?)

BASE=/mnt/disks/sdbt/zooscannet/ZooScanSet/imgs
OUT=data

classes=""

while read -d , p; do
    classes=$classes" $p"
done < selected_categories.txt

for dir in $classes; do
   echo "$dir"
   mkdir -p "$OUT/$dir"

   ls "$BASE/$dir"| head -5000 | while read f; do 
          convert -resize 299x299 "$BASE/$dir/$f" -background white -gravity center -extent 299x299 "$OUT/$dir/$f"
   done
done
