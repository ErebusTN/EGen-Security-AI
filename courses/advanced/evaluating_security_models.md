# Evaluating AI Models for Security Applications

This course covers essential techniques for evaluating AI models in security contexts, focusing on specialized metrics, testing methodologies, and verification approaches that go beyond standard machine learning evaluation practices.

## Table of Contents

1. [Introduction to Security Model Evaluation](#introduction-to-security-model-evaluation)
2. [Security-Specific Evaluation Metrics](#security-specific-evaluation-metrics)
3. [Testing Methodologies for Security Models](#testing-methodologies-for-security-models)
4. [Evaluating Model Robustness](#evaluating-model-robustness)
5. [Red Team Testing](#red-team-testing)
6. [Evaluating False Positives and False Negatives](#evaluating-false-positives-and-false-negatives)
7. [Performance Evaluation Under Different Conditions](#performance-evaluation-under-different-conditions)
8. [Long-term Monitoring and Drift Detection](#long-term-monitoring-and-drift-detection)
9. [Interpreting and Communicating Results](#interpreting-and-communicating-results)
10. [Case Studies](#case-studies)

## Introduction to Security Model Evaluation

Traditional ML evaluation focuses on accuracy, precision, recall, and F1 scores. While these remain important, security applications have unique requirements:

- **Asymmetric costs**: False negatives (missing a threat) are often far more costly than false positives
- **Adversarial context**: Attackers actively work to evade detection
- **Evolving threats**: What works today may not work tomorrow
- **Operational requirements**: Performance under resource constraints matters

### Why Standard Evaluation Is Insufficient

Standard ML evaluation approaches often fail in security contexts because:

1. They assume stationary data distributions
2. They don't account for adversarial manipulation
3. They rarely consider detection latency
4. They emphasize average performance over worst-case scenarios

### A Framework for Security Model Evaluation

A comprehensive security model evaluation should include:

- Standard ML metrics (precision, recall, F1)
- Security-specific metrics
- Robustness testing against adversarial inputs
- Latency and resource utilization analysis
- Testing against novel threats
- Stability over time

## Security-Specific Evaluation Metrics

### Detection Rate Metrics

- **True Positive Rate (TPR)**: Percentage of actual threats correctly identified
- **False Positive Rate (FPR)**: Percentage of benign items incorrectly flagged
- **False Negative Rate (FNR)**: Percentage of threats missed
- **Alert Precision**: Percentage of alerts that correspond to actual threats

### Example: Calculating Security Metrics

```python
def calculate_security_metrics(y_true, y_pred, y_scores=None):
    """Calculate security-focused evaluation metrics."""
    # Basic classification metrics
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Security-specific metrics
    metrics = {
        'true_positive_rate': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'threat_detection_ratio': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'false_alarm_ratio': fp / (fp + tp) if (fp + tp) > 0 else 0,
        'miss_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
    }
    
    # Add AUC if scores are provided
    if y_scores is not None:
        metrics['auc'] = roc_auc_score(y_true, y_scores)
        
    return metrics
```

### Time-Based Metrics

For security systems, time matters:

- **Mean Time to Detect (MTTD)**: Average time from threat appearance to detection
- **Detection Latency**: Time required to process and classify a potential threat
- **Time to Alert (TTA)**: Total time from threat appearance to alert generation

### Resource Utilization Metrics

Security models must be resource-efficient:

- **CPU/Memory Usage**: Resource requirements during inference
- **Throughput**: Number of samples processed per second
- **Scalability**: Performance under increased load

## Testing Methodologies for Security Models

### Dataset Preparation for Security Evaluation

Security model evaluation requires carefully constructed datasets:

```python
def prepare_security_evaluation_dataset(benign_data, malicious_data, ratio=0.1):
    """Create a balanced evaluation dataset with realistic threat ratios."""
    # Set the ratio of malicious to benign (typically low in real-world)
    num_malicious = int(len(benign_data) * ratio / (1 - ratio))
    
    # Sample from malicious data to match desired ratio
    malicious_sample = random.sample(malicious_data, min(num_malicious, len(malicious_data)))
    
    # Combine datasets
    eval_data = benign_data + malicious_sample
    labels = [0] * len(benign_data) + [1] * len(malicious_sample)
    
    # Shuffle while keeping data and labels aligned
    combined = list(zip(eval_data, labels))
    random.shuffle(combined)
    eval_data, labels = zip(*combined)
    
    return list(eval_data), list(labels)
```

### Cross-Validation for Security Models

Security data often exhibits temporal patterns. Standard random cross-validation may not be appropriate:

```python
def temporal_cross_validation(data, labels, timestamps, n_folds=5):
    """Perform temporal cross-validation to simulate real-world evaluation."""
    # Sort data by timestamp
    sorted_indices = np.argsort(timestamps)
    sorted_data = [data[i] for i in sorted_indices]
    sorted_labels = [labels[i] for i in sorted_indices]
    
    # Create temporal folds
    fold_size = len(sorted_data) // n_folds
    folds = []
    
    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(sorted_data)
        
        test_indices = list(range(start_idx, end_idx))
        train_indices = list(range(0, start_idx))
        
        # Train only on data that came before test data
        folds.append((train_indices, test_indices))
    
    return folds
```

### Stratified Sampling for Security Evaluation

Ensure your test set includes different threat categories:

```python
def stratified_security_sampling(data, threat_categories, test_size=0.2):
    """Create stratified samples ensuring all threat categories are represented."""
    train_indices = []
    test_indices = []
    
    # Group indices by threat category
    category_indices = {cat: [] for cat in set(threat_categories)}
    for i, category in enumerate(threat_categories):
        category_indices[category].append(i)
    
    # For each category, split indices into train and test
    for category, indices in category_indices.items():
        n_test = max(1, int(len(indices) * test_size))
        
        # Randomly select test indices
        category_test_indices = random.sample(indices, n_test)
        category_train_indices = [i for i in indices if i not in category_test_indices]
        
        train_indices.extend(category_train_indices)
        test_indices.extend(category_test_indices)
    
    return train_indices, test_indices
```

## Evaluating Model Robustness

### Adversarial Testing

Test your model against adversarial examples:

```python
def evaluate_adversarial_robustness(model, test_data, test_labels, attack_functions):
    """Evaluate model performance against various adversarial attacks."""
    base_accuracy = model.evaluate(test_data, test_labels)
    
    robustness_results = {
        'base_accuracy': base_accuracy,
        'attacks': {}
    }
    
    for attack_name, attack_fn in attack_functions.items():
        # Generate adversarial examples
        adv_examples = attack_fn(model, test_data, test_labels)
        
        # Evaluate model on adversarial examples
        adv_accuracy = model.evaluate(adv_examples, test_labels)
        
        # Calculate robustness score (ratio of adversarial to base accuracy)
        robustness_score = adv_accuracy / base_accuracy
        
        robustness_results['attacks'][attack_name] = {
            'accuracy': adv_accuracy,
            'robustness_score': robustness_score
        }
    
    return robustness_results
```

### Concept Drift Detection

Security threats evolve over time. Evaluate your model's ability to handle drift:

```python
def evaluate_concept_drift_resilience(model, time_series_data, time_series_labels, time_windows):
    """Evaluate model performance across different time periods to detect concept drift."""
    drift_results = {}
    
    for window_name, (start_time, end_time) in time_windows.items():
        # Select data in the time window
        window_indices = [i for i, t in enumerate(time_series_data['timestamp']) 
                          if start_time <= t < end_time]
        
        window_data = [time_series_data['features'][i] for i in window_indices]
        window_labels = [time_series_labels[i] for i in window_indices]
        
        # Evaluate model on this time window
        window_metrics = calculate_security_metrics(window_labels, model.predict(window_data))
        drift_results[window_name] = window_metrics
    
    # Calculate stability metrics across windows
    stability = calculate_performance_stability(drift_results)
    
    return {
        'window_results': drift_results,
        'stability_metrics': stability
    }
```

### Noise and Variation Testing

Real-world security data contains noise. Test your model's resilience:

```python
def test_noise_resilience(model, test_data, test_labels, noise_levels=[0.05, 0.1, 0.2]):
    """Test model performance with varying levels of input noise."""
    noise_results = {
        'base': calculate_security_metrics(test_labels, model.predict(test_data))
    }
    
    for noise_level in noise_levels:
        # Add noise to test data
        noisy_data = add_noise_to_features(test_data, noise_level)
        
        # Evaluate on noisy data
        predictions = model.predict(noisy_data)
        metrics = calculate_security_metrics(test_labels, predictions)
        
        noise_results[f'noise_{noise_level}'] = metrics
    
    return noise_results
```

## Red Team Testing

Red team testing involves simulating real attackers to challenge your model:

### Creating a Red Team

1. **Assemble a diverse team**: Include security experts, ML engineers, and domain specialists
2. **Define objectives**: What specific aspects of the model need testing?
3. **Set rules of engagement**: Determine what techniques are allowed
4. **Establish success criteria**: When is an evasion considered successful?

### Red Team Testing Procedure

```python
def red_team_evaluation(model, red_team_samples, success_criteria):
    """Evaluate model against red team attack samples."""
    results = {
        'total_samples': len(red_team_samples),
        'evasion_attempts': [],
        'successful_evasions': 0,
        'evasion_rate': 0.0
    }
    
    for sample in red_team_samples:
        # Get model prediction
        prediction = model.predict(sample['input'])
        
        # Check if evasion was successful based on criteria
        success = success_criteria(prediction, sample['true_label'])
        
        # Record results
        results['evasion_attempts'].append({
            'sample_id': sample['id'],
            'technique': sample['technique'],
            'success': success,
            'model_confidence': prediction['confidence']
        })
        
        if success:
            results['successful_evasions'] += 1
    
    # Calculate evasion rate
    results['evasion_rate'] = results['successful_evasions'] / results['total_samples']
    
    return results
```

### Documenting Red Team Findings

Document all successful evasion techniques in a structured format:

```python
def document_evasion_technique(technique_name, description, inputs, success_rate):
    """Document a successful evasion technique."""
    return {
        'name': technique_name,
        'description': description,
        'example_inputs': inputs,
        'success_rate': success_rate,
        'date_discovered': datetime.now().isoformat(),
        'mitigation_status': 'Unmitigated',
        'priority': classify_evasion_priority(success_rate)
    }
```

## Evaluating False Positives and False Negatives

In security contexts, not all errors are equal:

### Cost-Sensitive Evaluation

```python
def cost_sensitive_evaluation(y_true, y_pred, fp_cost=1, fn_cost=10):
    """Evaluate predictions with custom costs for FP and FN."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate total cost
    total_cost = (fp * fp_cost) + (fn * fn_cost)
    
    # Calculate cost per sample
    n_samples = len(y_true)
    cost_per_sample = total_cost / n_samples
    
    return {
        'total_cost': total_cost,
        'cost_per_sample': cost_per_sample,
        'fp_total_cost': fp * fp_cost,
        'fn_total_cost': fn * fn_cost,
        'confusion_matrix': {
            'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
        }
    }
```

### Threshold Optimization for Security

In security applications, the optimal classification threshold may not be 0.5:

```python
def optimize_security_threshold(model, validation_data, validation_labels, 
                                fp_cost=1, fn_cost=10, granularity=0.01):
    """Find optimal threshold to minimize security-weighted cost."""
    # Get prediction scores
    scores = model.predict_proba(validation_data)[:, 1]
    
    best_threshold = 0.5  # Default
    best_cost = float('inf')
    
    # Try different thresholds
    thresholds = np.arange(0, 1 + granularity, granularity)
    results = []
    
    for threshold in thresholds:
        # Apply threshold
        predictions = (scores >= threshold).astype(int)
        
        # Calculate cost
        cost_result = cost_sensitive_evaluation(
            validation_labels, predictions, fp_cost, fn_cost
        )
        
        results.append({
            'threshold': threshold,
            'cost': cost_result['total_cost'],
            'fp': cost_result['confusion_matrix']['fp'],
            'fn': cost_result['confusion_matrix']['fn'],
            'tp': cost_result['confusion_matrix']['tp'],
            'tn': cost_result['confusion_matrix']['tn']
        })
        
        # Update best threshold
        if cost_result['total_cost'] < best_cost:
            best_cost = cost_result['total_cost']
            best_threshold = threshold
    
    return {
        'best_threshold': best_threshold,
        'best_cost': best_cost,
        'threshold_results': results
    }
```

### False Positive Analysis

Analyze false positives to identify patterns:

```python
def analyze_false_positives(model, fp_examples, cluster_method='dbscan'):
    """Analyze false positives to identify clusters and patterns."""
    # Extract features from false positive examples
    fp_features = extract_features(fp_examples)
    
    # Cluster false positives
    if cluster_method == 'dbscan':
        clustering = DBSCAN(eps=0.3, min_samples=5).fit(fp_features)
    else:
        clustering = KMeans(n_clusters=5).fit(fp_features)
    
    cluster_labels = clustering.labels_
    
    # Analyze clusters
    clusters = {}
    for i, cluster_id in enumerate(cluster_labels):
        if cluster_id == -1:  # Noise in DBSCAN
            continue
            
        if cluster_id not in clusters:
            clusters[cluster_id] = []
            
        clusters[cluster_id].append(fp_examples[i])
    
    # Generate insights for each cluster
    cluster_insights = {}
    for cluster_id, examples in clusters.items():
        insights = generate_cluster_insights(examples)
        cluster_insights[cluster_id] = insights
    
    return {
        'num_clusters': len(clusters),
        'cluster_sizes': {k: len(v) for k, v in clusters.items()},
        'cluster_insights': cluster_insights
    }
```

## Performance Evaluation Under Different Conditions

Security models must work in various conditions:

### Stress Testing

```python
def stress_test_model(model, test_data, rates, duration_seconds=60):
    """Test model performance under different input rates."""
    results = {}
    
    for rate in rates:  # requests per second
        # Calculate batch size
        batch_size = min(1000, max(1, int(rate)))
        
        # Prepare test data
        num_batches = int(rate * duration_seconds / batch_size)
        
        # Initialize metrics
        latencies = []
        throughput_actual = 0
        successful = 0
        failed = 0
        
        # Run test
        start_time = time.time()
        for i in range(num_batches):
            batch_start_idx = (i * batch_size) % len(test_data)
            batch_end_idx = min(batch_start_idx + batch_size, len(test_data))
            batch = test_data[batch_start_idx:batch_end_idx]
            
            # Process batch and measure latency
            batch_start = time.time()
            try:
                _ = model.predict(batch)
                successful += len(batch)
                latency = time.time() - batch_start
                latencies.append(latency)
            except Exception:
                failed += len(batch)
            
            # Control rate
            elapsed = time.time() - start_time
            expected_elapsed = (i + 1) / (rate / batch_size)
            if expected_elapsed > elapsed:
                time.sleep(expected_elapsed - elapsed)
        
        # Calculate metrics
        test_duration = time.time() - start_time
        throughput_actual = successful / test_duration
        
        results[f'rate_{rate}'] = {
            'target_rate': rate,
            'actual_throughput': throughput_actual,
            'success_rate': successful / (successful + failed) if (successful + failed) > 0 else 0,
            'avg_latency': np.mean(latencies) if latencies else float('inf'),
            'p95_latency': np.percentile(latencies, 95) if latencies else float('inf'),
            'p99_latency': np.percentile(latencies, 99) if latencies else float('inf'),
            'max_latency': max(latencies) if latencies else float('inf')
        }
    
    return results
```

### Resource Utilization Analysis

```python
def analyze_resource_utilization(model, test_data, batch_sizes=[1, 10, 100, 1000]):
    """Analyze CPU, memory and throughput for different batch sizes."""
    results = {}
    
    for batch_size in batch_sizes:
        # Measure resource utilization
        cpu_usage = []
        memory_usage = []
        inference_times = []
        
        # Process test data in batches
        num_batches = math.ceil(len(test_data) / batch_size)
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(test_data))
            batch = test_data[start_idx:end_idx]
            
            # Measure CPU and memory before
            process = psutil.Process(os.getpid())
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Process batch and measure time
            start_time = time.time()
            _ = model.predict(batch)
            inference_time = time.time() - start_time
            
            # Measure CPU and memory after
            cpu_percent = process.cpu_percent()
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            
            # Record measurements
            cpu_usage.append(cpu_percent)
            memory_usage.append(mem_after - mem_before)
            inference_times.append(inference_time)
        
        # Calculate metrics
        total_samples = len(test_data)
        total_time = sum(inference_times)
        throughput = total_samples / total_time if total_time > 0 else 0
        
        results[f'batch_{batch_size}'] = {
            'batch_size': batch_size,
            'avg_cpu_percent': np.mean(cpu_usage),
            'max_cpu_percent': max(cpu_usage),
            'avg_memory_increase_mb': np.mean(memory_usage),
            'throughput_samples_per_sec': throughput,
            'avg_latency_sec': np.mean(inference_times),
            'latency_per_sample_ms': np.mean(inference_times) * 1000 / batch_size
        }
    
    return results
```

## Long-term Monitoring and Drift Detection

Monitor your model's performance over time:

### Drift Detection

```python
def detect_model_drift(reference_data, current_data, drift_threshold=0.05):
    """Detect if model inputs or outputs are drifting over time."""
    # Calculate distribution statistics for reference and current data
    ref_stats = calculate_distribution_stats(reference_data)
    current_stats = calculate_distribution_stats(current_data)
    
    # Calculate feature-wise distribution distances
    feature_drift = {}
    for feature in ref_stats:
        # Calculate KL divergence or another statistical distance
        if feature in current_stats:
            distance = calculate_distribution_distance(
                ref_stats[feature], current_stats[feature]
            )
            feature_drift[feature] = distance
    
    # Identify drifting features
    drifting_features = {
        feature: distance for feature, distance in feature_drift.items()
        if distance > drift_threshold
    }
    
    return {
        'drift_detected': len(drifting_features) > 0,
        'num_drifting_features': len(drifting_features),
        'drifting_features': drifting_features,
        'all_feature_drift': feature_drift
    }
```

### Performance Monitoring Over Time

```python
def monitor_model_performance(model, monitoring_data_stream, evaluation_interval=86400):
    """Continuously monitor model performance over time."""
    performance_history = []
    last_evaluation_time = time.time()
    
    # Collect data for initial evaluation
    current_batch = []
    
    for data_point, label in monitoring_data_stream:
        # Add to current batch
        current_batch.append((data_point, label))
        
        # Check if it's time to evaluate
        current_time = time.time()
        if current_time - last_evaluation_time >= evaluation_interval and current_batch:
            # Extract features and labels
            features = [x[0] for x in current_batch]
            labels = [x[1] for x in current_batch]
            
            # Evaluate model
            predictions = model.predict(features)
            metrics = calculate_security_metrics(labels, predictions)
            
            # Record performance
            performance_history.append({
                'timestamp': current_time,
                'metrics': metrics,
                'num_samples': len(current_batch)
            })
            
            # Reset for next interval
            last_evaluation_time = current_time
            current_batch = []
    
    return performance_history
```

## Interpreting and Communicating Results

Presenting results effectively is crucial:

### Security Dashboard Metrics

```python
def generate_security_dashboard_metrics(model_evaluation_results):
    """Generate metrics for a security model dashboard."""
    return {
        'threat_detection': {
            'overall_detection_rate': model_evaluation_results['true_positive_rate'],
            'false_alarm_rate': model_evaluation_results['false_positive_rate'],
            'missed_detection_rate': model_evaluation_results['false_negative_rate']
        },
        'performance': {
            'average_processing_time': model_evaluation_results['avg_latency'],
            'samples_per_second': model_evaluation_results['throughput'],
            'resource_utilization': model_evaluation_results['avg_cpu_percent']
        },
        'robustness': {
            'drift_status': 'Stable' if not model_evaluation_results.get('drift_detected', False) else 'Drifting',
            'adversarial_robustness': model_evaluation_results.get('robustness_score', 0),
            'stability_score': model_evaluation_results.get('stability', 0)
        },
        'security_risk': calculate_security_risk_score(model_evaluation_results)
    }
```

### Management Reporting

Create reports for different stakeholders:

```python
def generate_executive_summary(model_evaluation_results):
    """Generate an executive summary of model evaluation results."""
    # Calculate key metrics
    detection_rate = model_evaluation_results['true_positive_rate'] * 100
    false_alarm_rate = model_evaluation_results['false_positive_rate'] * 100
    missed_threats = model_evaluation_results['false_negative_rate'] * 100
    
    # Determine overall rating
    if detection_rate > 95 and false_alarm_rate < 1:
        overall_rating = "Excellent"
    elif detection_rate > 90 and false_alarm_rate < 5:
        overall_rating = "Good"
    elif detection_rate > 80 and false_alarm_rate < 10:
        overall_rating = "Satisfactory"
    else:
        overall_rating = "Needs Improvement"
    
    # Generate summary text
    summary = f"""
    Executive Summary: Security Model Evaluation
    
    Overall Rating: {overall_rating}
    
    Key Metrics:
    - Threat Detection Rate: {detection_rate:.1f}%
    - False Alarm Rate: {false_alarm_rate:.1f}%
    - Missed Threats: {missed_threats:.1f}%
    
    Risk Assessment:
    - The current model {'meets' if overall_rating in ['Excellent', 'Good'] else 'does not meet'} security requirements.
    - {'No' if missed_threats < 5 else 'Some'} critical threats were missed during evaluation.
    - Resource utilization is {model_evaluation_results.get('avg_cpu_percent', 0):.1f}% of available capacity.
    
    Recommendations:
    {generate_recommendations(model_evaluation_results)}
    """
    
    return summary.strip()
```

## Case Studies

### Case Study 1: Evaluating a Phishing Detection Model

A financial institution deployed a transformer-based model to detect phishing attempts:

- **Initial accuracy**: 97% on test set
- **Production performance**: 92% detection rate with high false positives
- **Issue**: Train/test datasets didn't reflect seasonal variations in attacks
- **Solution**: Implemented time-series cross-validation and threshold optimization

### Case Study 2: Malware Detection Model Drift

A security vendor observed declining performance in their malware detection model:

- **Cause**: Adversaries developed new obfuscation techniques
- **Detection**: Monitoring showed increasing false negative rate
- **Solution**: Implemented drift detection and automated retraining

### Case Study 3: API Security Model Stress Testing

A cloud provider's API protection model failed during a major sales event:

- **Issue**: Model performed well in testing but failed under high load
- **Analysis**: Stress testing revealed performance degradation at 10x normal traffic
- **Solution**: Optimized inference pipeline and implemented batch processing

## Additional Resources

- [Google's Model Cards for Model Reporting](https://modelcards.withgoogle.com/about)
- [MITRE ATLAS (Adversarial Threat Landscape for AI Systems)](https://atlas.mitre.org/)
- [NIST AI Risk Management Framework](https://www.nist.gov/itl/ai-risk-management-framework)
- [MLSec Project](https://mlsec.org/)
- [Security Evaluation of Machine Learning Models: Stanford CS 259D](https://seclab.stanford.edu/CS259D/)

---

© 2023 EGen Security AI Team 