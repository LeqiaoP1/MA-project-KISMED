
# purpose

This is desgin contract for the video signals like TIR-ROI.

# Video input preprocessing

Applied to the Entire Video Clip: Before BEFORE patch division, the raw video clip as a whole undergoes standard input preprocessing (z-score) to prepare the input patches for the network.


# Per-patch target normalization (only for masked cubes for loss computation)

Local Patch Normalization: For the masked cubes that the decoder needs to reconstruct, it should calculate a per-cube mean and variance locally within each individual patch/cube.


Ground-truth pixel values inside each masked cube are standardized using that cube's local mean and variance.


The decoder's predicted pixels are compared against these locally normalized target pixels using Mean Squared Error (MSE) loss.

# Unmasked cubes 
do not undergo local per-patch normalization, but they do receive clip-level input normalization.

