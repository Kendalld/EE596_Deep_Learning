"""
Simple test script to verify the implementation works correctly.
"""

import sys
from pathlib import Path
import torch
import numpy as np

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import using absolute imports from src package
import importlib.util
spec = importlib.util.spec_from_file_location("data_preprocessing", src_path / "data_preprocessing.py")
data_preprocessing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_preprocessing)

spec = importlib.util.spec_from_file_location("mlp_mo", src_path / "mlp_mo.py")
mlp_mo = importlib.util.module_from_spec(spec)
sys.modules['mlp_mo'] = mlp_mo
spec.loader.exec_module(mlp_mo)

spec = importlib.util.spec_from_file_location("cnn_mo", src_path / "cnn_mo.py")
cnn_mo = importlib.util.module_from_spec(spec)
sys.modules['cnn_mo'] = cnn_mo
spec.loader.exec_module(cnn_mo)

spec = importlib.util.spec_from_file_location("training", src_path / "training.py")
training = importlib.util.module_from_spec(spec)
sys.modules['training'] = training
spec.loader.exec_module(training)

spec = importlib.util.spec_from_file_location("evaluation", src_path / "evaluation.py")
evaluation = importlib.util.module_from_spec(spec)
sys.modules['evaluation'] = evaluation
spec.loader.exec_module(evaluation)

# Now import the functions
from data_preprocessing import create_multi_output_labels, normalize_traces
from mlp_mo import MLP_MO, MultiOutputLoss, compute_branch_accuracy
from cnn_mo import CNN_MO
from training import PowerTraceDataset, train_mlp_mo, train_cnn_mo
from evaluation import evaluate_model

def test_data_preprocessing():
    """Test data preprocessing functions."""
    print("Testing data preprocessing...")
    
    # Create synthetic data
    n_traces = 100
    trace_length = 700
    traces = np.random.randn(n_traces, trace_length)
    plaintexts = np.random.randint(0, 256, n_traces)
    
    # Test label creation
    labels = create_multi_output_labels(plaintexts)
    assert labels.shape == (n_traces, 256), f"Expected shape (100, 256), got {labels.shape}"
    assert labels.dtype == np.int64, f"Expected int64, got {labels.dtype}"
    print("  ✓ Multi-output label creation works")
    
    # Test normalization
    normalized, params = normalize_traces(traces, method='standard')
    assert normalized.shape == traces.shape, "Normalization should preserve shape"
    print("  ✓ Trace normalization works")
    
    print("Data preprocessing tests passed!\n")


def test_mlp_mo():
    """Test MLP_MO model."""
    print("Testing MLP_MO model...")
    
    trace_length = 700
    batch_size = 16
    
    # Test Non-SoSL
    model = MLP_MO(trace_length=trace_length, shared_layer_size=0)
    x = torch.randn(batch_size, trace_length)
    output = model(x)
    assert output.shape == (batch_size, 256, 2), f"Expected shape ({batch_size}, 256, 2), got {output.shape}"
    print("  ✓ Non-SoSL MLP_MO forward pass works")
    
    # Test SoSL-200
    model = MLP_MO(trace_length=trace_length, shared_layer_size=200)
    output = model(x)
    assert output.shape == (batch_size, 256, 2), f"Expected shape ({batch_size}, 256, 2), got {output.shape}"
    print("  ✓ SoSL-200 MLP_MO forward pass works")
    
    # Test loss function
    labels = torch.randint(0, 2, (batch_size, 256))
    criterion = MultiOutputLoss()
    loss = criterion(output, labels)
    assert loss.item() > 0, "Loss should be positive"
    print("  ✓ Multi-output loss function works")
    
    # Test accuracy computation
    accuracies = compute_branch_accuracy(output, labels)
    assert accuracies.shape == (256,), f"Expected shape (256,), got {accuracies.shape}"
    print("  ✓ Branch accuracy computation works")
    
    print("MLP_MO tests passed!\n")


def test_cnn_mo():
    """Test CNN_MO model."""
    print("Testing CNN_MO model...")
    
    trace_length = 480
    batch_size = 16
    
    model = CNN_MO(trace_length=trace_length)
    x = torch.randn(batch_size, trace_length)
    output = model(x)
    assert output.shape == (batch_size, 256, 2), f"Expected shape ({batch_size}, 256, 2), got {output.shape}"
    print("  ✓ CNN_MO forward pass works")
    
    print("CNN_MO tests passed!\n")


def test_training():
    """Test training functions with synthetic data."""
    print("Testing training functions...")
    
    # Create synthetic dataset
    n_traces = 200
    trace_length = 700
    traces = np.random.randn(n_traces, trace_length)
    plaintexts = np.random.randint(0, 256, n_traces)
    labels = create_multi_output_labels(plaintexts)
    
    # Normalize
    traces, _ = normalize_traces(traces, method='standard')
    
    # Split
    n_train = 160
    train_traces = traces[:n_train]
    train_labels = labels[:n_train]
    val_traces = traces[n_train:]
    val_labels = labels[n_train:]
    
    # Create datasets
    train_dataset = PowerTraceDataset(train_traces, train_labels)
    val_dataset = PowerTraceDataset(val_traces, val_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Test MLP_MO training (short run)
    model = MLP_MO(trace_length=trace_length, shared_layer_size=200)
    device = 'cpu'
    
    print("  Running short training (5 epochs)...")
    history = train_mlp_mo(model, train_loader, val_loader, num_epochs=5,
                          device=device, correct_key=None)
    
    assert 'attack_time' in history, "History should contain attack_time"
    assert len(history['train_loss']) == 5, "Should have 5 training epochs"
    print("  ✓ MLP_MO training works")
    
    # Test evaluation
    metrics = evaluate_model(model, val_loader, device=device, correct_key=None)
    assert 'mean_accuracy' in metrics, "Metrics should contain mean_accuracy"
    print("  ✓ Model evaluation works")
    
    print("Training tests passed!\n")


def main():
    """Run all tests."""
    print("="*60)
    print("Testing Multi-Output SCA Implementation")
    print("="*60 + "\n")
    
    try:
        test_data_preprocessing()
        test_mlp_mo()
        test_cnn_mo()
        test_training()
        
        print("="*60)
        print("All tests passed! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

