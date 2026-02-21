# Instructor's Guide

## Why we teach this lesson

Deep learning has revolutionized materials science by enabling accelerated discovery, property prediction, and inverse design. This workshop bridges the gap between AI fundamentals and their practical applications in materials research.

## Intended Learning Outcomes

1. Participants understand how neural networks work
2. Can implement and train models using modern frameworks
3. Know how to prepare materials data for deep learning
4. Can evaluate and improve model performance
5. Understand domain-specific considerations (crystal structures, compositional features)

## Timing (2 days, ~8 hours total)

### Day 1 (4 hours)
- Introduction & setup (30 min)
- Neural network fundamentals (1 hour)
- Building first model (1 hour)
- Materials data preparation (1 hour)
- Break & Q&A (30 min)

### Day 2 (4 hours)
- Advanced architectures (1.5 hours)
- Transfer learning (1 hour)
- Model evaluation & troubleshooting (1 hour)
- Case studies & open discussion (30 min)

## Preparing Exercises

**Before the workshop:**
1. Create a sample materials dataset (CSV format)
2. Pre-download model weights (ResNet50, etc.) if no internet
3. Test all notebooks in the expected environment
4. Prepare Jupyter kernels

**Day before:**
- Send participants the setup instructions
- Verify all dependencies are installed
- Have backup USB with pre-built environment

## Other Practical Aspects

- **Hardware**: GPU access recommended but CPU works for small datasets
- **Internet**: Required for downloading pre-trained models (can pre-cache)
- **IDE**: Jupyter notebooks work best for interactive learning
- **Group size**: Optimal with 15-25 participants for support

## Interesting Questions You Might Get

1. **"How much data do I need?"** → Depends on complexity; 1000+ samples typically needed
2. **"Should I use TensorFlow or PyTorch?"** → Both are great; TensorFlow easier for beginners
3. **"How do I handle small datasets?"** → Data augmentation, transfer learning, regularization
4. **"Can this replace DFT calculations?"** → Complements, not replaces; good for screening
5. **"What about interpretability?"** → SHAP, attention mechanisms, saliency maps

## Typical Pitfalls

1. **Data leakage** - Test set features into training normalization
2. **Overfitting** - Not using validation sets or regularization
3. **Imbalanced data** - Training bias toward majority class
4. **Poor preprocessing** - Forgetting to normalize/scale features
5. **Unrealistic expectations** - Expecting 100% accuracy
6. **Not splitting by material** - Test set contains same materials as train
7. **Ignoring domain knowledge** - Not using feat features relevant to materials

## Tips for Engagement

- Encourage participants to experiment with hyperparameters
- Provide pre-trained checkpoints to save time
- Have real materials datasets ready
- Connect each lesson to actual papers/applications
- Celebrate errors as learning opportunities

