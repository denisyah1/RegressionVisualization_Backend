# RegressionVisualization_Backend
Backend files for my porto project regeression visualization

### Plot Data Storage Design

This project uses an in-memory plot store to temporarily hold
regression visualization data.

This approach is suitable for:
- single-user workflows
- local development
- portfolio demonstration

In a production environment, this component can be replaced
with Redis or database-backed storage without changing the API contract.
