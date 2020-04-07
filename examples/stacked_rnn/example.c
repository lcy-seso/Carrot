void foo(int M, int N, int P, int Q) {
  float Z[M + P];
  float A[M + P][N + Q];
  float B[N][M];
  float X[N];
  float Y[Q];

#pragma scop
  for (int i = 0; i < M; ++i) {
    Z[i] = 0;
    for (int j = 0; j < N; ++j) {
      Z[i] += (A[i][j] + B[j][i]) * X[j];
    }
    for (int k = 0; k < P; ++k) {
      for (int l = 0; l < Q; ++l) {
        Z[k] += A[k][l] * Y[l];
      }
    }
  }
#pragma endscop
}
