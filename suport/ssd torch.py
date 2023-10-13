import torch

def ssd(img1, img2):
  """Computes the sum of squared differences (SSD) between two images."""
  if img1.shape != img2.shape:
    raise ValueError("The two images must have the same shape.")

  ssd = torch.sum((img1 - img2)**2)
  return ssd

if __name__ == "__main__":
  img1 = torch.rand(1, 3, 100, 100)
  img2 = torch.rand(1, 3, 100, 100)
  ssd = ssd(img1, img2)
  print(ssd)