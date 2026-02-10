import torch
from sklearn.cluster import kmeans_plusplus

device = torch.device(0 if torch.cuda.is_available() else 'cpu')


def RMeans(X, fixedCenters, k, a, b, iters, seed, device):

  b = torch.tensor(b).to(device).view(-1, 1)


  centroids, _ = kmeans_plusplus(X, n_clusters=k, random_state=seed)


  centroids = torch.from_numpy(centroids).to(device)

  X = torch.from_numpy(X).to(device)

  fixedCenters = torch.tensor(fixedCenters, dtype=float).to(device)


  p = fixedCenters.shape[0]

  fixedCentersAvg = fixedCenters.sum(0, keepdims=True)/p

  eye_mat = torch.eye(k+p, device = device)

  firstTerms = []
  secondTerms = []

  obj = []

  with torch.no_grad():
    for iter in range(iters):
      
      allCentroids = torch.cat((centroids, fixedCenters), 0)

      allCentroids_X_Diff = allCentroids.unsqueeze(0) - X.unsqueeze(1)

      X_allCentroids_SDist = allCentroids_X_Diff.pow(2).sum(2)

      xLabels = X_allCentroids_SDist.argmin(1)

      xOHELabels = eye_mat[xLabels].double()

      binBelongnessOfXToAllClusters =  xOHELabels.T

      binBelongnessOfXToNewClusters = binBelongnessOfXToAllClusters[:k, :]

      clusterSizes = binBelongnessOfXToNewClusters.sum(1, keepdims=True)

      sse = X_allCentroids_SDist.min(1).values.sum()

      fixedCentersAvg_Centroids_Diff = fixedCentersAvg.unsqueeze(0)-centroids.unsqueeze(1)
      Centroids_fixedCentersAvg_SDist = fixedCentersAvg_Centroids_Diff.pow(2).sum(2)

      temp1 = a*sse
      temp2 = sum(b*Centroids_fixedCentersAvg_SDist)
      firstTerms.append(temp1)
      secondTerms.append(temp2)
      obj.append(temp1 - temp2)

      numerator = (a*torch.mm(binBelongnessOfXToNewClusters, X) - b*fixedCentersAvg)

      denominator = (a*clusterSizes - b)

      new_centroids = numerator/denominator

      centroids = torch.where(denominator>0, new_centroids, centroids)


  allCentroids = torch.cat((centroids, fixedCenters), 0)
  allCentroids_X_Diff = allCentroids.unsqueeze(0) - X.unsqueeze(1)
  X_allCentroids_SDist = allCentroids_X_Diff.pow(2).sum(2)
  inertia = X_allCentroids_SDist.min(1).values.sum()
  xLabels = X_allCentroids_SDist.argmin(1)



  return allCentroids, xLabels, inertia, firstTerms, secondTerms, obj
