class StandardScaler:
	def fit(self, X):
		self.mean_ = X.mean(axis=0)
		self.scale_ = X.std(axis=0)
		return self
	
	def transform(self, X):
		return (X - self.mean_) / self.scale_

	def fit_transform(self, X):
		return self.fit(X).transform(X)