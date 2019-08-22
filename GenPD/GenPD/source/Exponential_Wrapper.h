#pragma once
/*
class for matrix exponential calculation

*/
#include "KIOPS.h"
#include "math_headers.h"
#include "Eigen/Dense"
#include "unsupported/Eigen/MatrixFunctions"

struct expokit {

	/*Function expm:
		calculate the action of matrix exponential exp(t*A)*b
		Input: t--time
			A-- an abstract operator that overloads operator* returns the action of matrix A*b
			b-- the vector to be multiplied
	*/
	template <typename Real = float, typename Mat_type>
	static Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> expm(Real _in_t,  Mat_type& _in_A, const Eigen::Matrix<Real, Eigen::Dynamic, 1>& _in_b, Real anorm = 30000, int m = 30, Real tol = 1.0e-7) {
		using MatrixXR = Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>;
		auto n = _in_b.size();
		if (m > n) {
			m = n;
		}
		int mxrej = 10;  Real btol = Real(5.0e-7);
		Real gamma = Real(0.9); Real delta = Real(1.2);
		int mb = m; Real t_out = std::abs(_in_t);
		int nstep = 0; Real t_new = 0;
		Real t_now = 0; Real s_error = 0;
		Real rndoff = anorm * std::numeric_limits<Real>::epsilon();

		int k1 = 2;
		Real xm = Real(1.0) / m;
		Real normv = _in_b.norm();
		Real beta = normv;
		Real fact = (Real)std::pow((m + 1.0) / std::exp(1), m + 1)*std::sqrt(Real(2.0 * 3.14)*(m + 1));
		t_new = (Real)(1.0 / anorm)*std::pow(((fact*tol) / (4 * beta*anorm)), xm);
		Real t_new2 = Real((1.0 / anorm)*std::pow(((tol) / (4 * beta*anorm)), xm)*std::pow((m + 1) / (std::exp(1)), 1 + 1.f / m)*std::pow(std::sqrt(2 * 3.14*(m + 1)), 1.f / m));

		//Setting the maximum step as t
		t_new = _in_t;

		Real s = (Real)std::pow(10, (std::floor(std::log10(t_new)) - Real(1.0))); t_new = std::ceil(t_new / s)*s;
		Real sgn = _in_t / t_out;

		auto w = _in_b;
		//p:temporary variable to store matrix-vector multiplication
		auto p = w;
		Real hump = normv;

		MatrixXR V = Eigen::MatrixXf::Zero(n, m + 1);//MatrixXR(n, m + 1);
		MatrixXR H = Eigen::MatrixXf::Zero(m + 2, m + 2);
		//F=exp(tH)
		MatrixXR F;
		//H *= 0;
		//V *= 0;
		Real err_loc = 0;
		int mx = m;
		while (t_now < t_out) {
			nstep += 1;
			float t_step = t_new > (t_out - t_now) ? t_out - t_now : t_new;
			V.col(0) = (1.0 / beta)*w;
			//printf("current time: %f\n", t_now);
			//arnoldi iteration
			for (int j = 0; j < m; j++) {
				p = _in_A(V.col(j));
				//printf("Matrix Mult: %d\n",j);
				/*std::cout << p << std::endl;
				printf("V[:,j]:");
				std::cout << V.col(j) << std::endl;*/
				for (int i = 0; i < j + 1; i++) {
					H(i, j) = V.col(i).dot(p);
					p = p.eval() - H(i, j)*V.col(i);
				}//end loop i
				s = p.norm();
				if (s < btol) {
					k1 = 0;
					mb = j + 1;
					t_step = t_out - t_now;
					break;
				}
				H(j + 1, j) = s;
				V.col(j + 1) = (1 / s)*p;
			}//end loop j 0->m 
			//arnoldi iteration end
			Real avnorm = 0;
			if (k1 != 0) {
				H(m + 1, m + 0) = 1;
				avnorm = (_in_A(V.col(m))).norm();
			}

			int ireject = 0;
			//local error estimation begin
			while (ireject <= mxrej) {
				mx = mb + k1;
				//Exponential of the Core
				F = (sgn*t_step*H.topLeftCorner(mx, mx)).exp();
				/*std::cout << "first 10 V vector:\n";
				std::cout << V.block(0,0,10,10) << std::endl;
				std::cout << "expokit result:\n";
				std::cout << "H:\n";
				std::cout << H.block(0, 0, mx, mx) << std::endl;
				std::cout << "F:\n";
				std::cout << F << std::endl;*/

				if (k1 == 0) {
					err_loc = btol;
					break;
				}
				else {
					auto phi1 = std::abs(beta*F(m + 0, 0));
					auto phi2 = std::abs(beta*F(m + 1, 0) * avnorm);
					if (phi1 > 10 * phi2) {
						err_loc = phi2;
						xm = Real(1) / m;
					}
					else if (phi1 > phi2) {
						err_loc = (phi1*phi2) / (phi1 - phi2);
						xm = Real(1) / m;
					}
					else {
						err_loc = phi1;
						xm = Real(1) / (m - 1);
					}
				}
				//choose stepsize begin
				if (err_loc <= delta * t_step*tol) {
					break;
				}
				else {
					//printf("Local error too large: %f\n", err_loc);
					t_step = gamma * t_step * std::pow((t_step*tol / err_loc), xm);
					s = (Real)std::pow(10, (std::floor(std::log10(t_step)) - 1));
					t_step = std::ceil(t_step / s) * s;
					if (ireject == mxrej) {
						printf("The requested tolerance is too high.\n");
					}
					ireject = ireject + 1;
				}//end choose stepsize
			}//end local error estimate 
			//local error estimation end

			mx = mb + ((k1 - 1) > 0 ? (k1 - 1) : 0);
			w = V.topLeftCorner(n, mx)*(beta*F.topLeftCorner(mx, 1));
			beta = w.norm();
			hump = hump > beta ? hump : beta;
			t_now = t_now + t_step;

			t_new = gamma * t_step * std::pow((t_step*tol / err_loc), xm);
			s = (Real)std::pow(10, (std::floor(std::log10(t_new)) - 1));
			t_new = std::ceil(t_new / s) * s;

			err_loc = err_loc > rndoff ? err_loc : rndoff;
			s_error = s_error + err_loc;
			//t_now += t_step;
		}//end while t_now<t_out
		return w;

	}
};

struct ERE_wrapper {
	
	ERE_wrapper(SparseMatrix * _in_K, SparseMatrix * _in_D, SparseMatrix* _in_invM, const VectorX* _in_force, VectorX* _in_x, VectorX* _in_v){
		K = _in_K;
		D = _in_D;
		f = _in_force;
		x = _in_x;
		v = _in_v;
		invM = _in_invM;
		//get f subtracting the linear part:
		int n = (*x).size();
		c = VectorX::Zero( n);
		//tempf is a vector of length n
		VectorX tempf = (*invM)*(*f + (*K)*(*x));
		c = tempf;
		n_K_evals = 1;
	}
	
	void ERE_one_step(ScalarType dt) {
		
		int n = (*x).size();
#define use_kiops

#ifndef use_kiops
		VectorX xv1(2*n+1);
#else 
		Matrix xv1(2 * n + 1, 1);
#endif
		xv1.block(0,0,n,1) = *x;
		xv1.block(n,0,n,1) = *v;
		xv1(2 * n) = 1;
		VectorX result=xv1;

#ifndef use_kiops
		result = expokit::expm(dt, *this, xv1.eval(), 3000.f, 35, 1e-5f);//dt * (this->operator()(xv1.eval()));
#else
		int out_m;
		result = KIOPS::KIOPS(out_m, std::vector<ScalarType>{dt}, *this, xv1.eval(), ScalarType(1e-7));//dt * (this->operator()(xv1.eval()));
		//result =  expokit::expm(dt, *this, xv1.col(0).eval(), 3000.f, 35, 1e-5f);
		//printf("Difference:%f\n", (result - result2).norm());
		
#endif
		
		*x = result.block(0, 0, n, 1);
		*v = result.block(n, 0, n, 1);
		//printf("number of K evaluation: %d\n", n_K_evals);
	}


	VectorX operator()(const VectorX & _in)  {
		//matrix structure:
		/*
			   0        I          0  
			-inv(M)K -inv(M)D      c   
			   0        0          0     
			Multiply x                
			         v
					 z

			input: 
			result = v
			         -invMK*x-invMD*v + c*z
					 0
		*/
		int n = (*x).size();
		VectorX result(2 * n + 1);
		result.head(n) = _in.segment(n, n);
		if (D) {
			result.segment(n, n) = (-*invM)*(*K)*_in.head(n) - (*invM)*(*D)*_in.segment(n, n) + c * _in.tail(1);
		}
		else {
			//no damping
			n_K_evals++;
			result.segment(n, n) = (-*invM)*(*K)*_in.head(n) + c * _in.tail(1);
		}
		result(2*n) = ScalarType(0);
		
		return result;
	}

	//pointer to stiffness matrix
	SparseMatrix* K;
	//pointer to Damping matrix, by default NULL
	SparseMatrix* D;
	SparseMatrix* invM;

	const VectorX* f;
	VectorX* x;
	VectorX* v;
	VectorX c;
	
	int n_K_evals;
};

struct OIKD_wrapper {

	OIKD_wrapper(SparseMatrix * _in_K, SparseMatrix * _in_D, SparseMatrix* _in_invM, ScalarType dt) {
		K = _in_K;
		D = _in_D;
		invM = _in_invM;
		if (D) {
			//invMK = (-*invM)*(*K);
			//invMD = (-*invM)*(*D);
			invMK_row = (*invM)*(*K);
			invMD_row = (*invM)*(*D);
		}
		else {
			//invMK = (-*invM)*(*K);
			invMK_row = (*invM)*(*K);
		}
		_dt = dt;
	}

	VectorX operator()(const VectorX & _in) {
		//matrix structure:
		/*
			   0        dt*I          
			-dt*inv(M)K -dt*inv(M)D     
			                     
			Multiply x
					 v

			input: _in

			result = dt*v
					 -dt*invMK*x-dt*invMD*v 
		*/
		int n = (_in).size()/2;
		VectorX result(2*n);
		result.head(n) = _in.segment(n, n);

		if (D) {
			//result.segment(n, n) = (-*invM)*(*K)*_in.head(n) - (*invM)*(*D)*_in.segment(n, n);
			//result.segment(n, n) = invMK*_in.head(n) - invMD*_in.segment(n, n);
			result.segment(n, n) = -invMK_row *_in.head(n) - invMD_row*_in.segment(n, n);
		}
		else {
			//no damping
			//result.segment(n, n) = (-*invM)*(*K)*_in.head(n);
			//result.segment(n, n) = invMK*_in.head(n);
			result.segment(n, n) = -invMK_row *_in.head(n);
		}

		return _dt*result;
	}

	//pointer to stiffness matrix
	SparseMatrix* K;
	//pointer to Damping matrix, by default NULL
	SparseMatrix* D;
	SparseMatrix* invM;
	SparseMatrix invMD, invMK;
	Eigen::SparseMatrix<ScalarType, Eigen::RowMajor> invMD_row, invMK_row;
	ScalarType _dt;
};