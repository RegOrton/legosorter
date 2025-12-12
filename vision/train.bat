@echo off
echo Starting training in Docker container...
docker-compose run --rm vision python src/train.py --epochs 5 --limit 100
pause
