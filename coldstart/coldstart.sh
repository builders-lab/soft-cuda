cp coldstart.tmp coldstart.prev 2>/dev/null

git add ..

git commit -m "chore/ Running summarizer"

{
  echo "=== GIT LOG ==="
  git log --oneline -7
  echo "=== GIT DIFF ==="
  git diff HEAD
  echo "=== TODO ==="
  grep -r TODO ../src/ 2>/dev/null
  echo "=== PREVIOUS SESSION SUMMARY ==="
  [ -f coldstart.prev ] && cat coldstart.prev
} | python coldstart_agent.py > logs/${filename}

cat logs/${filename} > coldstart.tmp


git add ..

git commit -m "chore/ Summarizer completion logging"
