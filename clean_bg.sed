#!/usr/bin/sed -f

# Remove lines beginning with "Plale" or "Plate"
/^Plale/d
/^Plate/d

# Remove TEXT line and everything between TEXT and TRANSLATION
# Replace TRANSLATION line with <EOT>
/^TEXT/,/^TRANSLATION/{
  /^TEXT/d
  /^TRANSLATION/{
    s/.*/<EOT>/
    b
  }
  d
}
