package skripte;
class Aufgabe {
    public static void main(String[] args) {
        printIter(0, 100, new Phi());
        printIter(0, 100, new Newton());
    }

    static void printIter(double xnull, int n, Funktor f) {
        double x=xnull;
        for(int i=1; i<=n; ++i) {
            x = f.function(x);
            System.out.printf("%s Iteration %d:\t %.53f\n", f.getName(), i, x);
        }
    }

    static void printIterTex(double x, int n, Funktor f) {
        for(int i=1; i<=n; ++i) {
            x = f.function(x);
            System.out.printf("%s_%d &= %.6f\n", f.texName, i, x);
        }
    }
}

abstract class Funktor {
    String texName;
    String getName() {
        return this.getClass().getName();
    }
    abstract double function(double x);
}

class Phi extends Funktor {
    Phi() {this.texName = "x";}
    public double function(double x) {
        return Math.exp(-(x * x));
    }
}

class Newton extends Funktor {
    Newton() {this.texName = "y";}
    public double function(double x) {
        return x - (Math.exp(-x*x)-x)/(-2*Math.exp(-x*x)-1);
    }
}