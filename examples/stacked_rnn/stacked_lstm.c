void StackLSTM(int N, int M, int K) {
  int a[N][M];
  int b[N][K];
  int c[N][M][K];  // output array
#pragma scop
  for (int i = 0; i < N; ++i) {      // batch size
    for (int j = 0; j < M; ++j) {    // sequence length
      for (int k = 0; k < K; ++k) {  // depth
        if (k == 0) {
          if (j == 0) {
            c[i][j][k] = a[i][j] + b[i][k];
          } else {
            c[i][j][k] = a[i][j] + c[i][j - 1][k];
          }
        } else {
          if (j == 0) {
            c[i][j][k] = c[i][j][k - 1] + b[i][k];
          } else {
            c[i][j][k] = c[i][j][k - 1] + c[i][j - 1][k];
          }
        }
      }
    }
  }
#pragma endscop
}
