# Task 3: Activation Steering

This task applies activation steering to control strict vs loose entity boundaries.

Key scripts:
- `src/tasks/task3_steering/prepare_boundary_sets.py`
- `src/tasks/task3_steering/extract_activations.py`
- `src/tasks/task3_steering/compute_steering.py`
- `src/tasks/task3_steering/run_steering.py`
- `src/tasks/task3_steering/evaluate_boundaries.py`

Quick start:
```bash
python -m src.tasks.task3_steering.prepare_boundary_sets --sample_count 300
python -m src.tasks.task3_steering.extract_activations
python -m src.tasks.task3_steering.compute_steering
python -m src.tasks.task3_steering.run_steering --sample_count 200
python -m src.tasks.task3_steering.evaluate_boundaries
```
