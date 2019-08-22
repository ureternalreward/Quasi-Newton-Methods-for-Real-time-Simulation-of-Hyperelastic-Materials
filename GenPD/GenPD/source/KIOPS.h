#pragma once
/*% Evaluates a linear combinaton of the phi functions
% evaluated at tA acting on vectors from u, that is
%
% w(i) = phi_0(T(i) A) u(:, 1) + phi_1(T(i) A) u(:, 2) + phi_2(T(i) A) u(:, 3) + ...
%
% The size of the Krylov subspace is changed dynamically
% during the integration. The Krylov subspace is computed
% using the incomplete orthogonalization method.
%
% License : GNU LGPLv2.1
%
% REFERENCES :
% * Gaudreault, S., Rainwater, G. and Tokman, M., 2018. KIOPS: A fast adaptive Krylov subspace solver for exponential integrators. Journal of Computational Physics.
%
% Based on the PHIPM and EXPMVP codes (http://www1.maths.leeds.ac.uk/~jitse/software.html)
% * Niesen, J. and Wright, W.M., 2011. A Krylov subspace method for option pricing. SSRN 1799124
% * Niesen, J. and Wright, W.M., 2012. Algorithm 919: A Krylov subspace algorithm for evaluating the \varphi-functions appearing in exponential integrators. ACM Transactions on Mathematical Software (TOMS), 38(3), p.22
%
% PARAMETERS:
%   tau_out    - Array of [T(1), ..., T(end)]
%   A          - the matrix argument of the phi functions.
%   u          - the matrix with columns representing the vectors to be
%                multiplied by the phi functions.
%
% OPTIONAL PARAMETERS:
%   tol        - the convergence tolerance required.
%   m_init     - an estimate of the appropriate Krylov size.
%   mmin, mmax - let the Krylov size vary between mmin and mmax
%
% RETURNS:
%   w        - the linear combination of the phi functions
%              evaluated at tA acting on the vectors from u.
%   m        - the Krylov size of the last substep.
%   stats(1) - number of substeps
%   stats(2) - number of rejected steps
%   stats(3) - number of Krylov steps
%   stats(4) - number of matrix exponentials

% n is the size of the original problem
% p is the highest indice of the phi functions*/

#include "Eigen/Dense"
#include <vector>
#include "timer_wrapper.h"

//#define TIMING
namespace KIOPS {

	/*Function KIOPS
	*/
	template <typename Real = float, typename Mat_type>
	Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic> KIOPS(int & out_m,
		std::vector<Real> tau_out,
		Mat_type& A_in,
		const Eigen::Matrix<Real, Eigen::Dynamic, Eigen::Dynamic>& u_in,
		Real tol = 1.0e-16,
		int m_init = 10,
		int mmin = 4,
		int mmax = 120) {

		using Mat = Eigen::Matrix < Real, Eigen::Dynamic, Eigen::Dynamic >;
		//dimension
		auto n = u_in.rows();

		int ppo = u_in.cols();
		//p: number of additional vectors.
		int p = ppo - 1;

		//handle pure matrix exponential situation
		Mat u(n, ppo >= 2 ? ppo : 2);
		if (p == 0) {
			p = 1;
			u << u_in, 0 * u_in;
		}
		else {
			u = u_in;
		}

		int orth_len = mmax;

		//size of the Krylov subspace

		if (!(m_init > mmin)) {
			m_init = mmin;
		}
		int m = std::max(mmin, std::min(m_init, mmax));

		//preallocate matrix
		Mat V(n + p, mmax + 1), H(mmax + 1, mmax + 1);

		V.setZero();
		H.setZero();

		//some params
		int step = 0;
		int krystep = 0;
		int ireject = 0;
		int reject = 0;
		int exps = 0;
		int sgn = *tau_out.rbegin() > 0 ? 1 : -1;
		Real tau_now = 0;
		Real tau_end = std::abs(*tau_out.rbegin());
		int happy = 0;
		int j = 0;

		int numSteps = tau_out.size();

		//initial condition
		Mat w(n, numSteps);
		w.setZero();
		//augmented w as the tail
		Mat w_aug(p, 1);
		w.col(0) = u.col(0);

		Real mu = 1, nu = 1;

		// Normalization factors
		/*std::cout << "u:\n";
		std::cout << u << std::endl;*/
		Real normU = u.rightCols(p).lpNorm<1>();
		if (ppo > 1 && normU > 0) {
			Real ex = std::ceil(std::log2(normU));
			nu = Real(std::pow(2, -ex));
			mu = Real(std::pow(2, ex));
		}
		else {
			nu = 1;
			mu = 1;
		}

		// Flip the rest of the u matrix
		Mat u_flip = nu * u.rightCols(p).rowwise().reverse();

		// Compute and initial starting approximation for the step size
		Real tau = tau_end;

		// Setting the safety factors and tolerance requirements
		Real gamma = Real(0.2), gamma_mmax = Real(0.1);
		if (tau_end > 1) {
			gamma = Real(0.2);
			gamma_mmax = Real(0.1);
		}
		else {
			gamma = Real(0.9);
			gamma_mmax = Real(0.6);
		}
		Real delta = Real(1.4);

		// Used in the adaptive selection
		int oldm = 0;
		int m_new = 0;
		Real tau_new = 0;
		Real oldomega;
		Real order;
		Real oldtau = NAN;
		Real omega = NAN;
		Real orderold = 1;
		Real kest;
		Real kestold = 1;



		//l is the index of current time output
		int l = 0;

		//norm of w
		Real beta = 0;
		Real err = 0;
#ifdef TIMING
		TimerWrapper timer,timer_ortho,timer_exponential,timer_A, timer_uflip;
		timer.Tic();
		timer.Pause();
		timer_ortho.Tic();
		timer_ortho.Pause();
		timer_exponential.Tic();
		timer_exponential.Pause();
		timer_A.Tic();
		timer_A.Pause();
		timer_uflip.Tic();
		timer_uflip.Pause();
		int n_mat_mult = 0;
#endif
		while (tau_now < tau_end) {
			// Compute necessary starting information
			if (j == 0) {
				// Update the last part of w
				w_aug(p - 1) = mu;
				for (int k = 1; k < p; k++) {
					//updating index
					w_aug(p - 1 - k) = tau_now / k * w_aug(p - k);
				}


				// Initialize the matrices V and H
				H.setZero();

				// Normalize initial vector(this norm is nonzero)
				beta = std::sqrt(w.col(l).squaredNorm() + w_aug.squaredNorm());

				// The first Krylov basis vector
				V.block(0, 0, n, 1) = (1 / beta) * w.col(l);
				V.block(n, 0, p, 1) = (1 / beta) * w_aug;
			}//end j=0
			
			//incomplete orthogonalization process
			
			
			while (j < m) {
				// Augmented matrix - vector product
#ifdef TIMING
				timer.Resume();
				timer_A.Resume();
#endif
				V.block(0, j + 1, n, 1) = A_in(V.block(0, j, n, 1));
#ifdef TIMING
				timer_A.Pause();
				n_mat_mult++;
				
				timer_uflip.Resume();
#endif
				V.block(0, j + 1, n, 1) += u_flip * V.block(n, j, p, 1);
#ifdef TIMING
				timer_uflip.Pause();
				timer.Pause();
#endif
				V.block(n, j + 1, p - 1, 1) = V.block(n + 1, j, p - 1, 1).eval();
				V(n + p - 1, j + 1) = 0;

				//Modified Gram - Schmidt
#ifdef TIMING
				timer_ortho.Resume();
#endif
				for (int i = std::max(0, j - orth_len + 1); i < j + 1; i++) {
					H(i, j) = V.col(i).dot(V.col(j + 1));
					V.col(j + 1) -= H(i, j) * V.col(i);
				}
#ifdef TIMING
				timer_ortho.Pause();
#endif
				//end

				Real nrm = V.col(j + 1).norm();
				//printf("nrm:%f\n", nrm);
				// Happy breakdown
				if (nrm < tol) {
					happy = 1;
					//j represent the size of Krylov space
					j = j + 1;
					break;
				}

				H(j + 1, j) = nrm;
				V.col(j + 1) *= (1 / nrm);

				
				krystep = krystep + 1;
				j = j + 1;
			}//end j<m
			
			// To obtain the phi_1 function which is needed for error estimate
			H(0, j) = 1;
			// Save h_j + 1, j and remove it temporarily to compute the exponential of H
			Real nrm = H(j, j - 1);
			H(j, j - 1) = 0;

			// Compute the exponential of the augmented matrix
#ifdef TIMING
			timer_exponential.Resume();
#endif
			Mat F = (sgn * tau * H.block(0, 0, j + 1, j + 1)).exp();
#ifdef TIMING
			timer_exponential.Pause();
#endif
			/*std::cout << "\nKIOPS result:\n";
			std::cout << "first 10 V vector:\n";
			std::cout << V.block(0, 0, 10, 10) << std::endl;
			std::cout << "H:\n";
			std::cout << H.block(0, 0, j + 1, j + 1) << std::endl;
			std::cout << "F:\n";
			std::cout << F.block(0,0,(10>j?j+1:10),1) << std::endl;*/
			exps = exps + 1;

			// Restore the value of H_{ m + 1,m }
			H(j, j - 1) = nrm;

			if (happy) {
				// Happy breakdown; wrap up
				omega = 0;
				happy = 0;
				m_new = m;
				tau_new = std::min(tau_end - (tau_now + tau), tau);
			}
			else {
				//Krylov subspace is not exact
				// Local truncation error estimation
				err = std::abs(beta * nrm * F(j - 1, j));

				// Error for this step
				oldomega = omega;
				omega = tau_end * err / (tau * tol);
				
				//the conclusion is when omega is nan, we need a smaller time step size.
				/*std::cout << "F:\n";
				std::cout << F << std::endl;
				std::cout << "Hblock:\n";
				std::cout << H.block(0, 0, j + 1, j + 1) << std::endl;
				std::cout << "reduced timestep expH:" << std::endl;
				std::cout << (sgn * 0.2 *tau * H.block(0, 0, j + 1, j + 1)).exp() << std::endl;
				printf("omega:%f\n", omega);
				printf("tau_end:%f, err:%f, tau:%f, tol:%f\n", tau_end,err,tau,tol);*/
				// Estimate order
				if (m == oldm && tau != oldtau && ireject >= 1) {
					order = std::max(Real(1), std::log(omega / oldomega) / std::log(tau / oldtau));
					orderold = 0;
				}
				else if (orderold || ireject == 0) {
					orderold = 1;
					order = j / Real(4);
				}
				else {
					orderold = 1;
				}

				// Estimate k
				if (m != oldm && tau == oldtau && ireject >= 1) {
					kest = std::max(Real(1.1), std::pow(omega / oldomega, (1 / (oldm - m))));
					kestold = 0;
				}
				else if (kestold || ireject == 0) {
					kestold = 1;
					kest = 2;
				}
				else {
					kestold = 1;
				}

				Real remaining_time;
				if (omega > delta||isnan(omega)) {
					remaining_time = tau_end - tau_now;
				}
				else {
					remaining_time = tau_end - (tau_now + tau);
				}

				// Krylov adaptivity

				Real same_tau = std::min(remaining_time, tau);
				Real tau_opt = tau * std::pow((gamma / omega), (1 / order));
				tau_opt = std::min(remaining_time, std::max(tau / 5, std::min(5 * tau, tau_opt)));

				int m_opt = int(std::ceil(j + std::log(omega / gamma) / std::log(kest)));
				//clamp between mmin, mmax, 3/4*m, 4/3*m

				m_opt = int(std::max(mmin, std::min(mmax, std::max((int)std::floor(3.0 / 4.0 * m), std::min(m_opt, (int)std::ceil(4.0 / 3.0 * m))))));

				if (isnan(omega)) {
					m_opt = m;
 					tau_opt = tau / 5;
				}

				if (j == mmax) {
					if (omega > delta) {
						m_new = j;
						tau_new = tau * std::pow((gamma_mmax / omega), (1 / order));
						tau_new = std::min(tau_end - tau_now, std::max(tau / 5, tau_new));
					}
					else {
						tau_new = tau_opt;
						m_new = m;
					}
				}
				else {
					m_new = m_opt;
					tau_new = same_tau;
					if (isnan(omega)) {
						tau_new = tau / 5;
					}
				}//end j== mmax

			}//else happy ending

			//Check error against target
			if (omega <= delta) {
				// Yep, got the required tolerance; update
				reject = reject + ireject;
				step = step + 1;

				// Udate for tau_out in the interval(tau_now, tau_now + tau)
				int blownTs = 0;
				Real nextT = tau_now + tau;
				for (int k = l; k < numSteps; k++) {
					if (std::abs(tau_out[k]) < std::abs(nextT)) {
						blownTs = blownTs + 1;
					}
				}

				if (blownTs != 0) {
					// Copy current w to w we continue with.
					if (l + blownTs < w.cols()) {
						w.col(l + blownTs) = w.col(l);
					}

					for (int k = 0; k < blownTs; k++) {
						Real tauPhantom = tau_out[l + k] - tau_now;
						Mat F2 = (sgn * tauPhantom * H.block(0, 0, j, j)).exp();
						w.col(l + k) = beta * V.block(0, 0, n, j) * F2.block(0, 0, j, 1);
					}

					l = l + blownTs;
				}

				// Using the standard scheme
				w.col(l) = beta * V.block(0, 0, n, j)* F.block(0, 0, j, 1);

				// Update tau_out
				tau_now = tau_now + tau;

				j = 0;
				ireject = 0;
			}
			else {
				// Nope, try again
				ireject = ireject + 1;

				// Restore the original matrix
				H(0, j) = 0;

			}//end else omega<=delta

			oldtau = tau;
			tau = tau_new;

			oldm = m;
			m = m_new;
		}//end while tau_now<tau_end

		out_m = m;

		/*std::cout << "V:\n";
		std::cout << V << std::endl;

		std::cout << "H:\n";
		std::cout << H << std::endl;*/
#ifdef TIMING
		printf("number of multiplication = %d\n", n_mat_mult);
		timer_A.Toc();
		timer_A.Report("A*v");
		timer_uflip.Toc();
		timer_uflip.Report("uflip");
		timer.Toc();
		timer.Report("build subspace");
		timer_ortho.Toc();
		timer_ortho.Report("orthogonalization");
		timer_exponential.Toc();
		timer_exponential.Report("exponential");
#endif
		return w;
	}
}//end namespace KIOPS

