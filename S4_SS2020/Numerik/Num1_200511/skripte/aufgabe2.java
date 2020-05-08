package skripte;
import java.lang.Math;

class Main {
    static int getSingleP() {
        float x=0.5f;
        int p=0;
        while(1f + x != 1f) {
            x/=2;
            ++p;
        }
        return p;
    }
    static int getDoubleP() {
        double x=0.5;
        int p=0;
        while(1d + x != 1d) {
            x/=2;
            ++p;
        }
        return p;
    }
    static int getSingleR() {
        int e=1;
        int r=1;
        while(1f/(float) Math.pow(2, e) != 0) {
            e = 2*e+1;
            ++r;
        }
        return r;
    }
    static int getDoubleR() {
        int e=1;
        int r=1;
        while(1d/Math.pow(2, e) != 0) {
            e = 2*e+1;
            ++r;
        }
        return r;
    }
    public static void main(String[] args) {
        System.out.printf("Single: p=%d, r=%d\nDouble: p=%d, r=%d\n", getSingleP(), getSingleR(), getDoubleP(), getDoubleR());
    }
}