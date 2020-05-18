package skripte;

class Main {
    static double W(double a, double h) {
        return (f(a+h) - f(a-h))/(2*h);
    }
    static double f(double x) {
        return Math.sin(x);
    }
    static double f1(double x) {
        return Math.cos(x);
    }
    public static void main(String[] args) {
        double h[] = new double[9];
        h[0] = Math.pow(2, -10);
        h[1] = Math.pow(2, -13);
        h[2] = Math.pow(2, -15);
        h[3] = Math.pow(2, -16);
        h[4] = Math.pow(3, 1/3) * Math.pow(2, -52/3);
        h[5] = Math.pow(2, -17);
        h[6] = Math.pow(2, -18);
        h[7] = Math.pow(2, -20);
        h[8] = Math.pow(2, -23);
        for(int i=0; i<h.length; ++i) {
            System.out.printf("%.10g\n", f1(1) - W(1, h[i]));
        }
        System.out.printf("%g\n", Math.abs(f1(1)-W(1, h[4])) - Math.abs(f1(1)-W(1, h[5])));
    }

}