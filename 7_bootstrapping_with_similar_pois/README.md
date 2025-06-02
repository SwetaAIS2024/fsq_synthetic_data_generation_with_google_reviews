# Bootstrapping with Similar POIs

This folder contains scripts and documentation for generating synthetic check-ins for sparse POIs using patterns from similar POIs.

## Method Description

Bootstrapping with Similar POIs addresses data sparsity by identifying POIs with very few check-ins (sparse POIs) and transferring check-in patterns from similar, more popular POIs (donor POIs). Similarity can be defined using POI features such as name, category, location, or review content. For each sparse POI, the method finds the most similar donor POIs and generates synthetic check-ins by mimicking the user and temporal patterns observed at those donors. This approach assumes that similar POIs attract similar user behaviors, thus enriching the sparse POIs with plausible synthetic activity while preserving realistic patterns in the dataset.

**Validation:**
- Similarity is typically measured using metrics like cosine similarity on POI features.
- The method can be validated by comparing the check-in distributions of bootstrapped POIs to their donor POIs and by measuring the reduction in sparsity for low-activity POIs.
