# Final Project Research

## Past Projects:

1. Generate new drawings using GANs
   1. in 2020 -- pre stable diffusion
2. Implicit Regularizing Effects of Batch Normalization
   1. 2020 -- BN was invented in 2015
3. New Training Techniques / Loss functions to mimic BN
   1. 2019 -- 

## Topics:

1. Variational Auto Encoder
2. Deep Neural Network Compression
   1. https://doi.org/10.48550/arXiv.1510.00149
   2. https://doi.org/10.1016/j.neucom.2022.11.072
3. Deep learning for Signal Processing
   1. De-Noising
   2. Human Activity Recognition
   3. Domain Generalization
      1. https://arxiv.org/pdf/2103.03097.pdf
         1. Domain generalization for image recognition
      2. https://ieeexplore.ieee.org/document/10288574
         1. Applications in wireless communication

# Idea of Work:

I want to do a similar line of work as the deep learning based MMIMO beamforming for 5G paper from 2018. Though this is an older paper, it has content that is essential to modern antenna systems. 

# Talking with the TA

Speaking with TA Matthew Parker 11/13/23

## Questions:

1. I’m really concerned about the final project and worried about my grade in the class. I’m worried that I won’t be able to find a novel thing to work on, or that if I do, that I won’t be able to put in enough time to get something working and get results. Hoping that you could help with these concerns... 
   - I’m worried that I won’t be able to know when I have something novel enough for this project.
2. I haven’t been able to find many datasets for RF signals of the kind that I want to work with. There are some mathematical frame-works that could provide for a kind of unsupervised training. Would this be enough work for this kind of project?
3. In the topic of domain generalization, I know that you said that you’re not an ML guy, but I was hoping that you could point me in the right direction for kinds of research that I could be looking into to help find some direction to go for this.

## Answers:

- Knowing if a project is novel enough, as long as it’s not plagiarized it’s novel. Even if you are just building off someone and trying to improve on it from them and doing something different.
- Plenty of groups do traditional projects like simple image recognition for determining albumn genres from cover art.
- As long as you don’t copy stuff you’ll be fine. Not expecting a great new thing that no one’s done before
- some people do work based off their research, but not really crazy
- TA’s grade relatively easy.
- There have been instances where people took existing paper and did a report on paper and used the same figures 
- As long as you actually show some improvement, but not “I couldn’t find any improvement”
- Plan on trying numerous different things because there’s no gaurantee that it will work.
- “Hey we tried this and it didn’t work” but we tried this other thing and it did
- Can you apply what you’ve learned this semester to some new problem
- Because you can apply it to anything
- They understand that we only have a few weeks.
- If there’s a large enough dataset, then there’s a way to make a project out of it.
- “Threw it together that morning” tends to be graded more harshly. 
- Can’t judge while they work on things
  - won’t get super nitpicking questions.
- If you can form a dataset from a simulation that might be fine
- Domain generalization
  - options could throw in various ranomized parameters
  - More general the data, the better the model would be in the end
  - This could be an interesting focus for project
  - Here’s what other people have done, but we’re trying to a make a more generalized model.
- Replication first?
  - If they have a published model and you can run / use / start with it. Then get their model working
  - Once it’s working then you can improve on it as I see fit.
  - If they don’t have it published, then don’t spend time to get same results as them
  - So just try my own thing as see how it compares
  - Can try something new to solve the same thing. Even if it doesn’t work better new information is important.



# Deep Learning and its Applications to Signal and Information Processing



# Domain Generalization

## Motivation:

- existing ML methods for wireless systems have several limitations
  - difficulty generalize under distribution shifts
  - inability to continuously learn from different scenarios
  - inability to quickly adapt to unseen scenarios
- performance appears to be over-fitted to specific set of simulation settings
- performance of DNNs versus state of the art estimators is dependent on training for a known distribution type
- calls for development of new ML training algorithms and establishment of rigorous evaluation protocols
- most studied is covariate shift
- leverage wireless communication domain knowledge to tailor generalizable ML algorithms
- beyond the scope of this work:
  - domain adaptation
  - transfer learning
  - zero-shot learning
  - multi-task learning
  - test time training
- Generalization is ability of DNNs to generalize to unseen scenarios
- Robustness is the stability of DNNs performance under noise and adversarial examples

## Contributions

- Part 1
  - Define DG problem and present four types of distribution shifts
  - contrast Df to different out of scope methods
- Part 2
  - Summarize different ML methods for DG
  - Focus on DNN training steps
  - learning frameworks
- Part 3
  - Literature review on previous attempts to ML for DG
  - Channel decoding
  - beamforming
  - MIMO estimation
  - reconfigurable intelligent surfaces
- Part 4
  - main challenges for application of data-driven ML tech in wireless comms

## Background

### DG Problem Formulation

### Related Concepts to DG

- Most assume that source and target domains are the same

**Supervised Learning**

- assumes that training and test samples are independent and identically distributed
- Naturally assumes a single identical domain

**Multi-task Learning**

- trains a single model to perform multiple tasks
- assumes that target and source domains are the same, but they may change
- DG does not have test-data

**Transfer learning**

- finetuning is one technique of transfer learning
- transfer one learned model to a new domain
- requires target samples to finetune

**Zero-Shot Learning**

- interested in generalizing to label shifts
- related to heterogeneous DG

**Test-time adaptation/training**

- trains for two tasks: main task, self-supervision task
- requires testdata for updating model parameters

**Continual lifelong learning**

- learns a multiple domain task sequentially without forgetting 

## DG Methods

**Data Manipulation**

- raw input space or ltaent input space
- adding random noise or transform to data
- data generation using generative models

1. Data Generation
   - Use variational auto encoders
   - Use GANs
   - mix-up method
     1. interpolate between different values and shift
2. Data Augmentation
   - gives cheap way to augment training datasets
   - inflate dataset size by transforming existing samples preserving labels
   - geometric and color transformations
   - random erasing or permutation
   - geared toward computer vision
   - constellation preserving augmentation
   - Domain randomization
     1. create a variety of datasets stemming from randomized properties in simulated environments
   - adversarial data augmentation
     1. guides augmentation by enhancing diversity of the dataset

**Domain Randomization**

- generate new samples from simulated environments

**Adversarial data augmentation**

- Randomly changing situation suggests more deliberate approach could improve generalization
- Adversary making new situations 
- perturb data along the direction of greatest domain change (domain gradient)
- iteratively add adverbially perturbed samples to training set
- computationally expensive
- Physics aware data augmentation collapse data augmentation to scenarios that enjoy scaling invariance

### Representation Learning

1. domain-invariant representation learning
   - DNN learns features that are invariant across different domains
2. Feature Disentanglement
   - DNNs learn a function that maps data sample to a feature vector
   - decompose feature sapce into set of feature subspaces
   - When feature representation is decomposable into non-overlapping subsets, representation is disentangled

### Domain Invariant Representation Learning Methods:

**Kernel-based methods**

- Feature disentanglement
- maps samples to a feature space using kernel function
  - Radial basis function, Gaussian, and Laplacian etc.
- Support vector machines
  - DNN decompose a feature representation into one of multiple sub-features
- Learn a kernel function by minimizing the distribution discrepancy between all data samples in feature space
- Domain invariant component analysis

**Domain Adversarial Learning**

- train GAN to learn domain invariant features
  - Discriminator trained to learn domain invariant feature repreesentations for DG

## Topics:

- Channel Coding
  - Autoencoders
- Channel Decoding
- Channel Estimation
- Beamforming
- Data Detection
- Beam Prediction
- RIS-Aided Wireless Communications
- Edge Networks

## Lessons Learned:

1. Lack of DG algorithms for Wireless
2. One-Sided focus on end-to-end
3. Need for wireless DG benchmarks

## Further Research

1. Channel coding
   1. **109** uses auto-encoders
   2. **110** uses DNNs
   3. **111** uses diffusion models
   4. **112** developed a meta-learning benchmark for DG
2. Channel Decoding
   1. **113-115** use iterative turbo/LDPC decoders based on belief-propagation framework
   2. **11**8 DNN based decoders showed near-optimal performance
   3. **119** unrolled BP algorithm into a DNN
   4. 112 explored meta learning approaches
   5. **121-123** substitute components but not the structure of decoding structures with DNNs
3. Channel estimation
   1. **125-128** did not analyze the impact of distribution shifts
   2. **129** used iterative shrinkage thresholding algorithm with learnable parameters
   3. **130** hypernetwork learns weighting factors for three channel models optimized in different settings
   4. **131** classical tracking methods use bank of Kalman filters with Doppler estimation
   5. **132** adapted to changing conditions by varying pilot length learned with DNN
   6.  **133-135** channel estimation and approximate message passing algorithms substituted with DNNs
      1. Onsager correction was beneficial to DNNs
   7. **138** DNN adapts to different number of receive antennas between 8 and 128
4. Beam Forming
   1. **140, 141** ML methods for low-complexity beam weight designs
   2. **142** unfolded the weighted minimum mean-square error estimator so each iteration step was a DNN
   3. **143** showed that Reinforcement learning accurately estimates uplink beam matrix
   4. **144** provides a comprehensive review of ML-based beamforming methods
   5. **145** used LSTM networks instead of WMMSE algorithm
   6. **87** used MAML algorithm for adaptive beamforming
   7. **147** used DNN for feature extraction use across multiple domains
   8. **148** used self-supervised learning to map uplink channels without accessing labeled training datasets
5. RIS-Aided Wireless Communications
   1. **168** ML approaches for RIS-aided communications
