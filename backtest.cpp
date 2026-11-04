#include <bits/stdc++.h>
using namespace std;

struct Row {
    int w, c;
    double r;
    double p[4];
};

int main(int argc, char** argv) {
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    double cost = (argc > 1 ? atof(argv[1]) : 10.0) * 1e-4;
    string line;
    getline(cin, line);
    vector<Row> rows;
    while (getline(cin, line)) {
        double a[7];
        if (sscanf(line.c_str(), "%lf,%lf,%lf,%lf,%lf,%lf,%lf", a, a + 1, a + 2, a + 3, a + 4, a + 5, a + 6) == 7) {
            Row x;
            x.w = (int)a[0];
            x.c = (int)a[1];
            x.r = a[2];
            for (int k = 0; k < 4; k++) x.p[k] = a[3 + k];
            rows.push_back(x);
        }
    }
    sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.w != b.w ? a.w < b.w : a.c < b.c; });
    int nc = 0;
    for (auto& x : rows) nc = max(nc, x.c + 1);
    vector<int> weeks;
    for (auto& x : rows) if (weeks.empty() || weeks.back() != x.w) weeks.push_back(x.w);
    const char* names[5] = {"equal_weight", "invvol_trailing", "invvol_har", "invvol_hybrid", "invvol_ensemble"};
    cout << "strategy,ann_return,ann_vol,sharpe,max_drawdown,worst_week,avg_turnover,weeks\n";
    for (int s = 0; s < 5; s++) {
        vector<double> prev(nc, 0.0), net;
        vector<double> turn;
        for (int w : weeks) {
            auto lo = lower_bound(rows.begin(), rows.end(), w, [](const Row& a, int v) { return a.w < v; });
            auto hi = upper_bound(rows.begin(), rows.end(), w, [](int v, const Row& a) { return v < a.w; });
            if (hi - lo < 6) continue;
            vector<double> cur(nc, 0.0);
            double sum = 0;
            for (auto it = lo; it != hi; ++it) {
                double v = s == 0 ? 1.0 : 1.0 / it->p[s - 1];
                cur[it->c] = v;
                sum += v;
            }
            double ret = 0, t = 0;
            for (auto it = lo; it != hi; ++it) {
                cur[it->c] /= sum;
                ret += cur[it->c] * it->r;
            }
            for (int c = 0; c < nc; c++) t += fabs(cur[c] - prev[c]);
            prev = cur;
            net.push_back(ret - cost * t);
            turn.push_back(t);
        }
        int n = net.size();
        double mean = 0, lg = 0;
        for (double x : net) {
            mean += x;
            lg += log1p(x);
        }
        mean /= n;
        double var = 0;
        for (double x : net) var += (x - mean) * (x - mean);
        double sd = sqrt(var / (n - 1));
        double eq = 1, peak = 1, dd = 0, worst = 1e9, tt = 0;
        for (int i = 0; i < n; i++) {
            eq *= 1 + net[i];
            peak = max(peak, eq);
            dd = min(dd, eq / peak - 1);
            worst = min(worst, net[i]);
            tt += turn[i];
        }
        printf("%s,%.4f,%.4f,%.3f,%.4f,%.4f,%.3f,%d\n", names[s], exp(lg * 52.0 / n) - 1, sd * sqrt(52.0), mean / sd * sqrt(52.0), dd, worst, tt / n, n);
    }
    return 0;
}
