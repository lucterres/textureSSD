### High-Level Description of the Algorithm

1. **Data Preparation:**
   - Collect and preprocess a dataset of seismic images containing salt domes.
   - Annotate the images with salt body masks, focusing on samples where the salt occupies between 10% and 90% of the total area.

2. **Training the Variational Autoencoder (VAE):**
   - **Encoder Structure:** Build the encoder using a simple feedforward neural network with four stacked dense layers.
   - **Latent Space:** Encode the seismic image data into a latent space by learning the distribution of key features (e.g., salt boundaries).
   - **Decoder Structure:** Construct the decoder with five stacked dense layers to decode latent variables into salt body masks.
   - **Training Process:** Train the VAE using only the annotated salt masks, focusing on learning the boundaries and shapes of salt bodies.

3. **Generating New Salt Body Masks:**
   - **Latent Sampling:** During inference, sample latent variables \( z \) from the learned prior distribution \( \pi(z) \).
   - **Mask Generation:** Decode the sampled \( z \) using the VAE’s decoder to generate new salt body masks that align with the distribution of the training data.

4. **Contextual Data Augmentation:**
   - Use the generated salt body masks as context orientation for augmenting the seismic data.
   - Identify the boundaries between the salt body and surrounding rock. Everything within the boundary is treated as salt.

5. **Texture Synthesis:**
   - Apply a non-parametric texture synthesis algorithm to generate new seismic image samples.
   - Ensure the synthesized image preserves the distinct characteristics of seismic images with salt domes, maintaining realistic texture and context zones.

6. **Final Output:**
   - Produce new seismic images that combine the generated salt body masks with synthesized textures, enhancing the dataset for further training and analysis.