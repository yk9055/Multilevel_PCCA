//[[Rcpp::depends(RcppArmadillo, RcppDist)]]
#include <RcppArmadillo.h>
#include <RcppDist.h>
#include <chrono>
#include <Rcpp.h>
using namespace Rcpp;
using namespace arma;
using namespace std;


arma::mat soft_thresholding(arma::mat x,
                            double lambda,
                            arma::mat weights){
  return(sign(x) % arma::max(arma::mat(size(x), fill::zeros) , abs(x) - lambda * weights)) ;
  
} // x (which is a loading matrix) and weights have the same dimension


arma::vec soft_thresholding_vec(arma::vec x,
                                double lambda,
                                arma::vec weights){
  return(sign(x) % arma::max(arma::vec(size(x), fill::zeros) , abs(x) - lambda * weights)) ;
  
} // x (which is a loading vector) and weights have the same length

// [[Rcpp::export()]]
List pred_Z(arma::vec params,
            arma::mat X,
            arma::vec idVar,
            arma::vec uid,
            const int N,
            const int n,
            const int kb,
            const int kw
){
  const int p = X.n_cols ;
  
  arma::mat Wb = reshape(params.subvec(0, (p*kb) -1), p, kb) ;
  arma::mat Ww = reshape(params.subvec(p*kb, p*(kb+kw)-1), p, kw) ;
  arma::mat PSI_b = diagmat(params.subvec(p*(kb+kw), (p*(kb+kw+1))-1)) ;
  arma::mat PSI_w = diagmat(params.subvec(p*(kb+kw+1), params.size()-1)) ;
  
  
  arma::mat sigma_b = Wb*Wb.t() + PSI_b ;
  
  arma::mat sigma_w = Ww*Ww.t() + PSI_w ;
  arma::mat inv_sigma_w = inv(sigma_w) ;
  
  
  arma::mat Zb(N, kb) ; // subject (cluster) level scores
  arma::mat Zw(n, kw) ; // observation-level scores
  arma::mat ZZb(n, kb*kb) ;
  arma::mat ZZw(n, kw*kw) ;
  
  arma::vec sum_ZZb(kb) ;   
  arma::vec sum_ZZw(kw) ;   
  
  
  for(int i = 0 ; i < N ; ++i){ 
    uvec row_i = find(idVar == uid[i]) ;
    int ni = row_i.n_elem ;
    arma::mat Xi = X.rows(row_i) ; // (ni * p)
    arma::vec ones_ni(ni, fill::ones) ;
    arma::mat I_ni = diagmat(ones_ni) ;
    
    arma::colvec sum_Xij = arma::sum(Xi, 0).as_col() ; // (p * 1)
    
    arma::mat sigma_i_star = (inv_sigma_w - inv(sigma_w + ni*sigma_b)) / ni ;
    
    arma::mat A = inv_sigma_w - ni*sigma_i_star ;
    arma::vec Zbi = Wb.t()*(A)*(sum_Xij) ; // (kb * 1)
    Zb.row(i) = Zbi.as_row() ;
    
    arma::vec Ui = PSI_b.t()*(A)*(sum_Xij) ; // (p * 1)
    
    arma::vec Zwi_new = kron(I_ni, Ww).t() * (kron(I_ni, inv_sigma_w) -
      kron(ones_ni * ones_ni.t(), sigma_i_star)) * vectorise(Xi.t()) ;
    
    arma::mat Zwi = arma::reshape(Zwi_new, kw, ni).t() ; // (ni * kw)
    Zw.rows(row_i) = Zwi ; // (n * kw)
    
    arma::mat ZZbi = eye(kb, kb) - ni*Wb.t() * (A)*Wb + Zbi*Zbi.t() ; // (kb * kb)
    ZZb.rows(row_i) = repmat(  ZZbi.as_row(), ni, 1) ;
    
    
    for(int j = 0 ; j < ni ; ++j){
      
      arma::mat ZZwij = eye(kw, kw) - Ww.t()*(inv_sigma_w - sigma_i_star)*Ww + Zwi.row(j).t() * Zwi.row(j) ; // (kw * kw)
      
      ZZw.row(row_i(j)) = ZZwij.as_row() ; 
    }
    
  }
  
  
  for(int kb_i = 0; kb_i < kb; ++kb_i){
    sum_ZZb(kb_i) = accu(ZZb.col(kb_i * kb + kb_i)) ;
  }
  
  for(int kw_i = 0; kw_i < kw; ++kw_i){
    sum_ZZw(kw_i) = accu(ZZw.col(kw_i * kw + kw_i)) ;
  }
  
  List expZ ;
  expZ["Zb"] = Zb;
  expZ["Zw"] = Zw;
  expZ["sum_ZZb"] = sum_ZZb ;
  expZ["sum_ZZw"] = sum_ZZw ;
  return(expZ) ;
  
}


// [[Rcpp::export()]]
arma::vec mpCCA_EM_cpp(arma::vec params, 
                       arma::mat X,
                       arma::vec idVar, 
                       arma::vec uid,
                       const int N,
                       const int n,
                       const int kb, 
                       const int kw,
                       const int G,
                       arma::vec n_features,
                       arma::mat lambda_b,
                       arma::mat lambda_w,
                       arma::mat weights_b,
                       arma::mat weights_w,
                       const int method, 
                       const int coord
){ 
  
  const int p = X.n_cols ;

  // containers
  arma::mat Zb(N, kb) ; // subject (cluster) level scores
  arma::mat Zw(n, kw) ; // observation-level scores
  arma::mat ZZb(n, kb*kb) ;
  arma::mat ZZw(n, kw*kw) ;

  
  arma::mat Wb = reshape(params.subvec(0, (p*kb) -1), p, kb) ;
  arma::mat Wb_old = Wb ;
  arma::mat Ww = reshape(params.subvec(p*kb, p*(kb+kw)-1), p, kw) ;
  arma::mat Ww_old = Ww;
  arma::mat PSI_b = diagmat(params.subvec(p*(kb+kw), (p*(kb+kw+1))-1)) ;
  arma::mat PSI_w = diagmat(params.subvec(p*(kb+kw+1), params.size()-1)) ;
  
  arma::mat sigma_w = Ww*Ww.t() + PSI_w ;
  arma::mat inv_sigma_w = inv(sigma_w) ;
  
  arma::mat sigma_b = Wb*Wb.t() + PSI_b ;
  
  // -- E step
  
  // Intermediate matrices
  arma::mat Wb1(p, kb, fill::zeros) ;
  arma::mat Wb2(kb, kb, fill::zeros) ;
  arma::mat Ww1(p, kw, fill::zeros) ;
  arma::mat Ww2(kb, kw, fill::zeros) ;
  arma::mat Ww3(kw, kw, fill::zeros) ;
  arma::mat UU(p, p, fill::zeros) ;
  arma::mat PSI_w1(kb, p, fill::zeros) ;
  arma::mat PSI_w2(p, p, fill::zeros) ;
  arma::mat PSI_w3(kw, p, fill::zeros) ;
  arma::mat PSI_w4(p, kb, fill::zeros) ; 
  arma::mat PSI_w5(kw, kb, fill::zeros) ;
  
  
  for(int i = 0 ; i < N ; ++i){ 
    uvec row_i = find(idVar == uid[i]) ;
    int ni = row_i.n_elem ;
    arma::mat Xi = X.rows(row_i) ; // (ni * p)
    arma::vec ones_ni(ni, fill::ones) ;
    arma::mat I_ni = diagmat(ones_ni) ;
    
    arma::colvec sum_Xij = arma::sum(Xi, 0).as_col() ; // (p * 1)
    
    arma::mat sigma_i_star = (inv_sigma_w - inv(sigma_w + ni*sigma_b)) / ni ;
    
    arma::mat A = inv_sigma_w - ni*sigma_i_star ;
    arma::vec Zbi = Wb.t()*(A)*(sum_Xij) ; // (kb * 1)
    Zb.row(i) = Zbi.as_row() ;
    
    arma::vec Ui = PSI_b.t()*(A)*(sum_Xij) ; // (p * 1)
    
    arma::vec Zwi_new = kron(I_ni, Ww).t() * (kron(I_ni, inv_sigma_w) -
      kron(ones_ni * ones_ni.t(), sigma_i_star)) * vectorise(Xi.t()) ;
    
    arma::mat Zwi = arma::reshape(Zwi_new, kw, ni).t() ;// (ni * kw)
    Zw.rows(row_i) = Zwi ; // (n * kw)
    
    arma::mat ZZbi = eye(kb, kb) - ni*Wb.t() * (A)*Wb + Zbi*Zbi.t() ; // (kb * kb)
    arma::mat inv_ZZbi = inv_sympd(ZZbi) ;
    ZZb.rows(row_i) = repmat(  ZZbi.as_row(), ni, 1) ;
    
    arma::mat UUi = PSI_b - ni*PSI_b*(A)*PSI_b + Ui*Ui.t() ; // (p * p)
    arma::mat inv_UUi = inv_sympd(UUi) ;
    
    arma::mat ZbUi = -ni*Wb.t()*(A)*PSI_b + Zbi*Ui.t() ; // (kb * p)          
    
    
    arma::mat sum_ZZwij(kw, kw, fill::zeros) ;
    arma::mat sum_ZZwbij(kw, kb, fill::zeros) ;
    arma::mat sum_ZwUij(kw, p, fill::zeros) ;
    arma::mat sum_XZwij ;
    for(int j = 0 ; j < ni ; ++j){
      
      arma::mat ZZwij = eye(kw, kw) - Ww.t()*(inv_sigma_w - sigma_i_star)*Ww + Zwi.row(j).t() * Zwi.row(j) ; // (kw * kw)
      sum_ZZwij = sum_ZZwij + ZZwij ;
      ZZw.row(row_i(j)) = ZZwij.as_row() ; 
      
      arma::mat ZZwbij = -Ww.t()*(A)*Wb + Zwi.row(j).t()*Zbi.t() ; // (kw * kb)
      sum_ZZwbij = sum_ZZwbij + ZZwbij ;
      
      arma::mat ZwUij = -Ww.t()*(A)*PSI_b + Zwi.row(j).t()*Ui.t() ; // (kw * p)
      sum_ZwUij = sum_ZwUij + ZwUij ;
      
    }
    
    
    // -- M step
    Wb1 = Wb1 + Xi.t()*(ones_ni*Zbi.t()) - ni*ZbUi.t() - Ww*sum_ZZwbij ; // (p * kb)
    Wb2 = Wb2 + ni*ZZbi ; // (kb * kb)
    
    sum_XZwij = Xi.t()*Zwi ; // (p * kw)
    
    Ww1 = Ww1 + sum_XZwij - sum_ZwUij.t() ; // (p * kw)
    Ww2 = Ww2 - sum_ZZwbij.t() ; // (kb * kw)
    Ww3 = Ww3 + sum_ZZwij ; // (kw * kw)
    
    UU = UU + UUi ; // (p * p)
    
    
    PSI_w1 = PSI_w1 - Zbi*sum_Xij.t() + ni*ZbUi ; // (kb * p) 
    PSI_w2 = PSI_w2 - 2*sum_Xij*Ui.t() + ni*UUi ; // (p * p)
    PSI_w3 = PSI_w3 - sum_XZwij.t() + sum_ZwUij ; // (kw * p)
    PSI_w4 = PSI_w4 - Ww*sum_ZZwbij ; // (p * kb)
    PSI_w5 = PSI_w5 + sum_ZZwbij ;
    
  }
  
  if(method ==0){
  Wb = Wb1*inv(Wb2) ;
  Ww = (Ww1 + Wb*Ww2)*inv(Ww3) ;
  }else{ // method = 1 for adaptive lasso estimates
    arma::vec PSI_w_diag = diagvec(PSI_w) ; 


    arma::vec sum_ZZb(kb) ;   
    arma::vec sum_ZZw(kw) ;   


    if(coord == 0){
      Wb = Wb1*inv(Wb2) ;
      for(int g = 0 ; g < G ; ++g){ // different lambda for different datasets & latent variables
        for(int kb_i = 0; kb_i < kb ; ++kb_i){
          if(g==0){ 
            Wb.rows(0,n_features(0)-1).col(kb_i) = soft_thresholding(Wb.rows(0,n_features(0)-1).col(kb_i),lambda_b(g, kb_i), weights_b.rows(0,n_features(0)-1).col(kb_i)) ;
          }else{
            Wb.rows(sum(n_features.subvec(0,g-1)),sum(n_features.subvec(0,g))-1).col(kb_i) = soft_thresholding(Wb.rows(sum(n_features.subvec(0,g-1)),sum(n_features.subvec(0,g))-1).col(kb_i),lambda_b(g, kb_i), weights_b.rows(sum(n_features.subvec(0,g-1)),sum(n_features.subvec(0,g))-1).col(kb_i)) ;
          }
        }
      }
    }else{

      // Coordinate ascent: feature-by-feature soft-thresholding (i.e., feature-by-feature and one coordinate at a time w.r.t latent variables)
      // For a given feature, update loadings sequentially from the first to kb latent variables
      arma::vec h(kb) ;
      for(int h_i = 0; h_i < kb; ++h_i){
        h(h_i) = h_i ;
      }

      double d = 1 ;
      int d_it = 0 ;
      while(d > 1e-7 & d_it < 1000){ 

        for(int kb_i = 0; kb_i < kb; ++kb_i){
          uvec h_ind = find( h != kb_i) ;

          arma::mat H(kb, kb, fill::zeros) ;
          H.col(kb_i).fill(1) ;
          H(kb_i,kb_i) = 0 ;
          uvec h_ind2 = find(H == 1) ;
          arma::vec sum_w_by_zz = sum( Wb.cols(h_ind) * ZZb.cols(h_ind2).t(), 1) ; // p*1 column vector
          sum_ZZb(kb_i) = accu(ZZb.col(kb_i * kb + kb_i)) ;
          arma::vec w_kb_i = (Wb1.col(kb_i) - sum_w_by_zz) /  sum_ZZb(kb_i) ;

          for(int g = 0 ; g < G ; ++g){ // different lambda for different datasets & latent variables
            if(g==0){
              w_kb_i.rows(0,n_features(0)-1) = soft_thresholding_vec(w_kb_i.rows(0,n_features(0)-1),lambda_b(g, kb_i), weights_b.rows(0,n_features(0)-1).col(kb_i)% PSI_w_diag.rows(0,n_features(0)-1) / sum_ZZb(kb_i)) ;
            }else{
              w_kb_i.rows(sum(n_features.subvec(0,g-1)), sum(n_features.subvec(0,g))-1) = soft_thresholding_vec(w_kb_i.rows(sum(n_features.subvec(0,g-1)),sum(n_features.subvec(0,g))-1),lambda_b(g, kb_i), weights_b.rows(sum(n_features.subvec(0,g-1)),sum(n_features.subvec(0,g))-1).col(kb_i)% PSI_w_diag.rows(sum(n_features.subvec(0,g-1)), sum(n_features.subvec(0,g))-1) / sum_ZZb(kb_i)) ;
            }
          }

          Wb.col(kb_i) = w_kb_i ;

        }
        d = sqrt(accu( square(Wb - Wb_old) )) ; 

        Wb_old = Wb ;

        d_it += 1 ;
      }
    }



    if(coord == 0){
      Ww = (Ww1 + Wb*Ww2)*inv(Ww3) ;

      for(int g = 0 ; g < G ; ++g){ // different lambda for different datasets & latent variables
        for(int kw_i = 0; kw_i < kw ; ++kw_i){
          if(g==0){
            Ww.rows(0,n_features(0)-1).col(kw_i) = soft_thresholding(Ww.rows(0,n_features(0)-1).col(kw_i),lambda_w(g, kw_i), weights_w.rows(0,n_features(0)-1).col(kw_i)) ;
          }else{ 
            Ww.rows(sum(n_features.subvec(0,g-1)),sum(n_features.subvec(0,g))-1).col(kw_i) = soft_thresholding(Ww.rows(sum(n_features.subvec(0,g-1)),sum(n_features.subvec(0,g))-1).col(kw_i),lambda_w(g, kw_i), weights_w.rows(sum(n_features.subvec(0,g-1)),sum(n_features.subvec(0,g))-1).col(kw_i)) ;
          }
        }
      }
    }else{
      // Coordinate ascent: feature-by-feature soft-thresholding (i.e., feature-by-feature and one coordinate at a time w.r.t latent varibles)
      // For a given feature, update loadings sequentially from the first to kw latent variables
      arma::mat Ww4 = Ww1 + Wb*Ww2 ; // A-B-C
      arma::vec h(kw) ;

      for(int h_i = 0; h_i < kw; ++h_i){
        h(h_i) = h_i ;
      }

      double d = 1 ;
      int d_it = 0 ;
      while(d > 1e-7 & d_it < 1000){ 

        for(int kw_i = 0; kw_i < kw; ++kw_i){
          uvec h_ind = find( h != kw_i) ;

          arma::mat H(kw, kw, fill::zeros) ;
          H.col(kw_i).fill(1) ;
          H(kw_i,kw_i) = 0 ;
          uvec h_ind2 = find(H == 1) ;
          arma::vec sum_ww_by_zzw = sum( Ww.cols(h_ind) * ZZw.cols(h_ind2).t(), 1) ; // p*1 column vector
          sum_ZZw(kw_i) = accu(ZZw.col(kw_i * kw + kw_i)) ;
          arma::vec w_kw_i = (Ww4.col(kw_i) - sum_ww_by_zzw) / sum_ZZw(kw_i)  ;

          for(int g = 0 ; g < G ; ++g){ // different lambda for different datasets & latent variables
            if(g==0){
              w_kw_i.rows(0,n_features(0)-1) = soft_thresholding_vec(w_kw_i.rows(0,n_features(0)-1),lambda_w(g, kw_i), weights_w.rows(0,n_features(0)-1).col(kw_i) % PSI_w_diag.rows(0,n_features(0)-1) / sum_ZZw(kw_i)) ;
            }else{
              w_kw_i.rows(sum(n_features.subvec(0,g-1)), sum(n_features.subvec(0,g))-1) = soft_thresholding_vec(w_kw_i.rows(sum(n_features.subvec(0,g-1)),sum(n_features.subvec(0,g))-1),lambda_w(g, kw_i), weights_w.rows(sum(n_features.subvec(0,g-1)),sum(n_features.subvec(0,g))-1).col(kw_i)% PSI_w_diag.rows(sum(n_features.subvec(0,g-1)), sum(n_features.subvec(0,g))-1) / sum_ZZw(kw_i)) ; 
            }
          }

          Ww.col(kw_i) = w_kw_i ;

        }
        d = sqrt(accu( square(Ww - Ww_old) )) ; 

        Ww_old = Ww ;

        d_it += 1 ;
      }

    }

  }

  PSI_b = diagmat(UU/N) ;
  
  PSI_w = diagmat( diagmat(arma::sum(X%X, 0)) + Wb*PSI_w1 + PSI_w2 +
    Ww*PSI_w3 + PSI_w4*Wb.t() + Wb*PSI_w5.t()*Ww.t()) / n ;
  
  arma::vec params_new = join_cols(vectorise(Wb), vectorise(Ww), PSI_b.diag(), PSI_w.diag()) ;
  params = params_new ;
  
  
  
  return(params_new) ;
}



// [[Rcpp::export()]]
double mpCCA_ll_sqEM_cpp(arma::vec params, 
                         arma::mat X,
                         arma::vec idVar, 
                         arma::vec uid,
                         const int N,
                         const int n,
                         const int kb, 
                         const int kw,
                         const int G,
                         arma::vec n_features,
                         arma::mat lambda_b,
                         arma::mat lambda_w,
                         arma::mat weights_b,
                         arma::mat weights_w,
                         const int method, 
                         const int coord){
  

  const int p = X.n_cols ;
  
  arma::mat Wb = reshape(params.subvec(0, (p*kb) -1), p, kb) ;
  arma::mat Ww = reshape(params.subvec(p*kb, p*(kb+kw)-1), p, kw) ;
  arma::mat PSI_b = diagmat(params.subvec(p*(kb+kw), (p*(kb+kw+1))-1)) ;
  arma::mat PSI_w = diagmat(params.subvec(p*(kb+kw+1), params.size()-1)) ;
  
  
  arma::mat sigma_w = Ww*Ww.t() + PSI_w ;
  arma::mat inv_sigma_w = inv(sigma_w) ;
  
  arma::mat sigma_b = Wb*Wb.t() + PSI_b ;
  
  // Evaluate convergence
  double ll = -0.5*n*log(2*datum::pi) ;
  for(int i = 0 ; i < N ; ++i){
    uvec row_i = find(idVar == uid[i]) ;
    int ni = row_i.n_elem ;
    arma::mat Xi = X.rows(row_i) ; // (ni * p)
    
    arma::vec ones_ni(ni, fill::ones) ;
    arma::mat I_ni = diagmat(ones_ni) ;
    arma::mat ones_ni_ni(ni, ni, fill::ones) ;
    
    arma::mat sigma_x_i = kron(I_ni, sigma_w) + kron(ones_ni_ni, sigma_b) ;
    arma::mat sigma_x_i_star = (inv_sigma_w - inv(sigma_w + ni*sigma_b))/ni ;
    arma::mat inv_sigma_x_i = kron(I_ni, inv_sigma_w) - kron(ones_ni_ni, sigma_x_i_star) ;
    
    arma::vec Xi_vec = vectorise(Xi, 1).as_col() ; // (nip * 1)
    arma::mat a = (Xi_vec.t()*inv_sigma_x_i*Xi_vec) ; //(1 * 1)
    ll = ll - 0.5*log(det(sigma_x_i)) - 0.5*a(0) ;
  }
  
  // penalized log-likelihood
  double P_lambda_b ;
  double P_lambda_w ;
  arma::mat weighted_Wb = weights_b % Wb ;
  arma::mat weighted_Ww = weights_w % Ww ;
  for(int g = 0 ; g < G ; ++g){ // different lambda for different datasets & latent vectors
    if(g==0){
      P_lambda_b = accu(lambda_b.row(g) % (sum(abs(weighted_Wb.submat(0,0,n_features(0)-1,kb-1)), 0))) ; // (1*kb) * (1*kb) elementwise mutiplication and then sum over kb elements // note: sum(x, 0) = colsums
      P_lambda_w = accu(lambda_w.row(g) % (sum(abs(weighted_Ww.submat(0,0,n_features(0)-1,kw-1)), 0))) ;
    }else{
      P_lambda_b = P_lambda_b + accu(lambda_b.row(g) % (sum(abs(weighted_Wb.submat(sum(n_features.subvec(0,g-1)),0,sum(n_features.subvec(0,g))-1,kb-1)), 0))) ;
      P_lambda_w = P_lambda_w + accu(lambda_w.row(g) % (sum(abs(weighted_Ww.submat(sum(n_features.subvec(0,g-1)),0,sum(n_features.subvec(0,g))-1,kw-1)), 0))) ;
    }
  }
  
  double pll = ll - P_lambda_b - P_lambda_w ;
  
  return(-pll) ;
}



// [[Rcpp::export()]]
List sparse_mpCCA_EM_cpp (arma::vec params,
                          arma::mat X,
                          arma::vec idVar,
                          arma::vec uid,
                          const int N,
                          const int n,
                          const int kb,
                          const int kw,
                          const double epsilon,
                          const int G,
                          arma::vec n_features,
                          arma::mat lambda_b,
                          arma::mat lambda_w,
                          arma::mat weights_b,
                          arma::mat weights_w,
                          const int maxit = 100,
                          const int method = 1,
                          const int coord = 1) {
  
  const int p = X.n_cols ;
  
  // ---- Set up
  // inputs
  double pll = 0;
  int it = 0 ;
  double tol = 1 ;


  arma::vec pll_it ;

  arma::vec params_new ;
  while((it < maxit) & (tol > epsilon)){
    double pll_old = pll ;

      params = mpCCA_EM_cpp(params,
                              X,
                              idVar,
                              uid,
                              N,
                              n,
                              kb,
                              kw,
                              G,
                              n_features,
                              lambda_b,
                              lambda_w,
                              weights_b,
                              weights_w,
                              method,
                              coord) ;


      pll = - mpCCA_ll_sqEM_cpp(params, 
                                X,
                                idVar,
                                uid,
                                N,
                                n,
                                kb,
                                kw,
                                G,
                                n_features,
                                lambda_b,
                                lambda_w,
                                weights_b,
                                weights_w,
                                method,
                                coord) ;

      Rcout << "Iteration " << it << "\r";
      it = it + 1 ;

      pll_it.resize(pll_it.size()+1);
      pll_it(pll_it.size()-1) = pll ;

      tol = max(abs(pll_old - pll)) ;
  }

  arma::mat Wb = reshape(params.subvec(0, (p*kb) -1), p, kb) ;
  arma::mat Ww = reshape(params.subvec(p*kb, p*(kb+kw)-1), p, kw) ;
  arma::mat PSI_b = diagmat(params.subvec(p*(kb+kw), (p*(kb+kw+1))-1)) ;
  arma::mat PSI_w = diagmat(params.subvec(p*(kb+kw+1), params.size()-1)) ;
  
  List predZ = pred_Z(params, X, idVar, uid, N, n, kb, kw);

  // returns
  List out ;
  out["Zb"] = predZ["Zb"] ; // subject (cluster) level scores
  out["Zw"] = predZ["Zw"] ; // observation-level scores
  out["Wb"] = Wb ; // between-subject loadings
  out["Ww"] = Ww ; // within-subject loadings
  out["PSI_b"] = PSI_b ; // residual variances at between-subject level
  out["PSI_w"] = PSI_w ; // residual variances at within-subject level
  out["pll"] = pll;
  out["pll_it"] = pll_it ;
  out["it"] = it ;
  out["lambda_b"] = lambda_b;
  out["lambda_w"] = lambda_w;
  out["sum_ZZb"] = predZ["sum_ZZb"] ;
  out["sum_ZZw"] = predZ["sum_ZZw"] ;

  return(out) ;

  }





