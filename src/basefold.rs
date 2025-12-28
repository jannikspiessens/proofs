use std::alloc::Global;
use std::cell::RefCell;
use itertools::{izip, Itertools};

use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{BigIntRing, IntegerRingStore};
use feanor_math::ring::{RingExtension, RingStore, RingBase, El};
use feanor_math::field::{Field, FieldStore};
use feanor_math::rings::finite::FiniteRing;
use feanor_math::rings::multivariate::{
    MultivariatePolyRing,
    MultivariatePolyRingStore,
    multivariate_impl::MultivariatePolyRingImpl
};
use feanor_math::rings::poly::{
    PolyRingStore,
    dense_poly::DensePolyRing
};

use crate::util::{CoeffRing, Coeff, FiatShamirSim};
use crate::codes::{
    LinearCode,
    foldablecodes::{FoldableCode, RSFoldableCode}
};
use crate::multilinear::{
    MultilinearBasis, MultilinearBasisEvals,
    get_hypercube_coeffs, evals_to_coeffs_inplace,
    evaluate_at_fromcoeff, evaluate_at_fromevals_inplace, evaluate_at_fromevals,
    sum_over_hypercube_withscalars, evalscalars_to_coeffscalars,
    sumcheck::{
        SumcheckBase, Sumcheck,
        sumcheck_sum,
        SCMultilinearIterator
    }
};


pub trait Proof {}

pub trait Commitment {}

pub trait MultilinearPCS {

    type Poly: MultivariatePolyRingStore<Type: MultivariatePolyRing>;
    type C: Commitment;
    type P: Proof;

    fn polyring(&self) -> &Self::Poly;

    fn coeffring(&self) -> &CoeffRing<Self::Poly> {
        self.polyring().get_ring().base_ring()
    }

    fn get_unipolyring(&self) -> DensePolyRing<CoeffRing<Self::Poly>>;

    fn get_challenge(&self) -> Coeff<Self::Poly>;

    fn commit(&self, poly: &[Coeff<Self::Poly>]) -> Self::C;
    
    fn open(&self, com: &Self::C, poly: &[Coeff<Self::Poly>]) -> bool;

    fn eval_slow(&self, com: &Self::C, z: &[Coeff<Self::Poly>],
        y: Coeff<Self::Poly>, poly: &El<Self::Poly>) -> Self::P;

    fn verify(&self, com: Self::C, z: &[Coeff<Self::Poly>],
        y: Coeff<Self::Poly>, poly: &[Coeff<Self::Poly>], proof: Self::P) -> bool;

    fn eval(&self, com: &Self::C, z: &[Coeff<Self::Poly>],
        y: Coeff<Self::Poly>, polycoeff: Option<&[Coeff<Self::Poly>]>,
        polyeval: Option<&[Coeff<Self::Poly>]>) -> Self::P;
}

pub struct BaseFoldPCS<'a, F, C>
    where F: RingStore<Type: Field>, C: FoldableCode<R = F>
{
 fs: RefCell<FiatShamirSim<'a, F>>,
    polyring: MultivariatePolyRingImpl<F>,
    code: C,
    ver_rep: usize,
    // time_efficient: bool // TODO: use this
}

impl<'a, F, C> BaseFoldPCS<'a, F, C>
    where F: RingStore<Type: Field>, C: FoldableCode<R = F>
{
    pub fn field(&self) -> &F {
        self.code.ring()
    }

    pub fn code(&self) -> &C {
        &self.code
    }

    pub fn proofsize(&self) -> usize {
        let d = self.code.d();
        let proofsize = d*2*self.ver_rep + 3*d +
            if self.code.k(0) == 0 { 0 } else { self.code.k(0).ilog2() as usize };
        let ZZbig: BigIntRing = BigIntRing::RING;
        let bits = ZZbig.abs_log2_ceil(&self.field().characteristic(ZZbig).unwrap()).unwrap();
        let authpath = self.code.n(d).ilog2() as usize * 256; // worst-case no pruning
        return proofsize*(bits + authpath)
    }
}

impl<'a, F, C> BaseFoldPCS<'a, F, C>
    where F: RingStore<Type: Field + FiniteRing>, C: FoldableCode<R = F>
{
    pub fn reset_fs(&self) {
        self.fs.borrow_mut().reset()
    }
}


impl<'a, R> BaseFoldPCS<'a, R, RSFoldableCode<'a, R>>
    where R: RingStore<Type: Field + FiniteRing> + Clone
{
    pub fn new(field: &'a R, varcount: usize, k0: usize, c: usize, ver_rep: usize) -> Self
    {
        assert!(k0.is_power_of_two());

        let code = RSFoldableCode::new(field, k0, c,
            varcount - (k0.ilog2() as usize));
        
        let polyring = MultivariatePolyRingImpl::new_with(field.clone(),
            varcount, 2*varcount as u16, (0, 0), Global);

        let fs = RefCell::new(FiatShamirSim::new(field));
        
        Self {
            fs,
            polyring, 
            code,
            ver_rep,
            // time_efficient
        }
    }

}

pub struct BaseFoldCommitment<R: RingStore>{
    // TODO: remove these pubs
    pub code_el: Vec<El<R>>
}
impl<R: RingStore> Commitment for BaseFoldCommitment<R>{}

pub struct BaseFoldProof<R: RingStore> {
    // TODO: remove these pubs
    pub code_els: Vec<Vec<El<R>>>,
    pub sumcheck_els: Vec<El<DensePolyRing<R>>>,
    pub sumcheck_last: Vec<El<R>>
}

impl<R: RingStore> BaseFoldProof<R> {
    pub fn clone(&self, ring: &R, polyring: &DensePolyRing<R>) -> Self {
        Self {
            code_els: self.code_els.iter().map(|v|
                v.iter().map(|el| ring.clone_el(el)).collect()).collect(),
            sumcheck_els: self.sumcheck_els.iter().map(|poly| polyring.clone_el(poly)).collect(),
            sumcheck_last: self.sumcheck_last.iter().map(|el| ring.clone_el(el)).collect()
        }
    }
}

impl<R: RingStore> Proof for BaseFoldProof<R>{}

impl<'a, C, R> MultilinearPCS for BaseFoldPCS<'a, R, C>
    where R: RingStore<Type: Field + FiniteRing> + Clone, C: FoldableCode<R = R>
{
    type Poly = MultivariatePolyRingImpl<R>;
    type C = BaseFoldCommitment<R>;
    type P = BaseFoldProof<R>;

    fn polyring(&self) -> &Self::Poly {
        &self.polyring
    }

    fn get_unipolyring(&self) -> DensePolyRing<CoeffRing<Self::Poly>> {
        DensePolyRing::new(self.coeffring().clone(), "X")
    }

    fn get_challenge(&self) -> Coeff<Self::Poly> {
        self.fs.borrow_mut().challenge()
    }

    fn commit(&self, poly: &[Coeff<Self::Poly>]) -> Self::C
    {
        BaseFoldCommitment{code_el: self.code.encode(poly) }
    }
    
    fn open(&self, _com: &Self::C, _poly: &[Coeff<Self::Poly>]) -> bool {
        // TODO
        true
    }

    fn eval_slow(&self, com: &Self::C, z: &[Coeff<Self::Poly>],
        _y: Coeff<Self::Poly>, poly: &El<Self::Poly>) -> Self::P
    {
        let d = self.code.d();
        let vc = self.polyring().indeterminate_count();
        assert!(z.len() == vc);
        let f = self.field();
        let unipolyring = self.get_unipolyring();

        let fsclone = self.fs.borrow().clone();

        let mut polys: Vec<El<DensePolyRing<R>>> = Vec::with_capacity(d);
        let eq = MultilinearBasis::new(f, &z).polynomial(&self.polyring);
        let mut wpoly = self.polyring.clone_el(poly);
        let mut scpoly = self.polyring.mul_ref_fst(poly, eq);
        //assert_el_eq!(self.coeffring(), sum_over_hypercube(&self.polyring, &scpoly, vc, &[]), y);
        let mut hunivar = sumcheck_sum(&self.polyring, &unipolyring, &scpoly, vc - 1);
        /*assert_el_eq!(self.coeffring(), &self.coeffring().add(
            unipolyring.evaluate(&hunivar, &self.coeffring().zero(), self.coeffring().identity()),
            unipolyring.evaluate(&hunivar, &self.coeffring().one(), self.coeffring().identity())
        ), &y);*/
        polys.push(hunivar);

        let mut proofcodes: Vec<Vec<El<R>>> = Vec::with_capacity(d);
        let mut topcode = &com.code_el;

        let mut last: Vec<El<R>> = Vec::with_capacity(
            if self.code.k(0) == 1 {0} else {self.code.k(0)});
        
        for dind in (0..d).rev() {
            let chall = self.get_challenge();
            let curfreevc = vc - (d - dind);
            let challconst = self.polyring.create_term(self.coeffring().clone_el(&chall),
                self.polyring.create_monomial((0..vc).map(|_| 0)));
            wpoly = self.polyring().specialize(&wpoly, curfreevc, &challconst);

            proofcodes.push(
                self.code.t(dind).enumerate().map(|(i, ti)| {
                    interpdeg1(f, ti, &topcode[i], &topcode[i + self.code.n(dind)], &chall)
                }).collect()
            );
            topcode = &proofcodes[d - 1 - dind];
            /*let wpolycode = self.code.encode(&get_hypercube_coeffs(&self.polyring,
                    &wpoly, curfreevc).iter().collect_vec());
            assert!((0..self.code.n(dind)).all(|i|
                self.coeffring().eq_el(&proofcodes[d-1-dind][i], &wpolycode[i])));*/

            if dind != 0 {
                scpoly = self.polyring().specialize(&scpoly, curfreevc, &challconst);
                //let tmpsum = unipolyring.evaluate(&polys[d - 1 - dind], &chall, self.coeffring().identity());
                hunivar = sumcheck_sum(&self.polyring, &unipolyring, &scpoly, curfreevc - 1);
                /*assert_el_eq!(self.coeffring(), &self.coeffring().add(
                    unipolyring.evaluate(&hunivar, &self.coeffring().zero(), self.coeffring().identity()),
                    unipolyring.evaluate(&hunivar, &self.coeffring().one(), self.coeffring().identity())
                ), &tmpsum);*/
                polys.push(hunivar);
            }
        }
    
        assert!(self.code.k(0).ilog2() as usize == vc - d);
        if self.code.k(0) > 1 {
            last = get_hypercube_coeffs(&self.polyring, &wpoly, self.code.k(0).ilog2() as usize);
        }

        // reset the FS to the state before eval so that verify can use it
        self.fs.replace(fsclone);

        BaseFoldProof {
            code_els: proofcodes,
            sumcheck_els: polys,
            sumcheck_last: last
        }
    }

    fn verify(&self, com: Self::C, z: &[Coeff<Self::Poly>],
        y: Coeff<Self::Poly>, poly: &[Coeff<Self::Poly>], proof: Self::P) -> bool
    {
        let d = self.code.d();

        if proof.code_els.len() != d || proof.sumcheck_els.len() != d ||
            (proof.sumcheck_last.len() > 0) == (self.code.k(0) == 1) {
            return false;
        }

        let mut challvec: Vec<Coeff<Self::Poly>> = Vec::with_capacity(d);

        let unipolyring = self.get_unipolyring();
        let mut tmp = y;

        let mut topcode = &com.code_el;
        let mut mu = (0..self.ver_rep).map(|_|
            rand::random_range(0..self.code.n(d-1))).collect_vec();

        self.open(&com, &poly) &&
        proof.code_els.iter().zip(proof.sumcheck_els.iter()).enumerate().all(|(i, (code, poly))| {
            let chall = self.get_challenge();

            let dind = d - 1 - i;
            let t = self.code.t(dind).collect_vec();
            let mut rescode = (0..self.ver_rep).all(|j| {
                let interp = interpdeg1(self.coeffring(), t[mu[j]],
                    &topcode[mu[j]], &topcode[mu[j] + self.code.n(dind)], &chall);
                self.coeffring().eq_el(&interp, &code[mu[j]])
            });
            if dind > 0 {
                mu.iter_mut().for_each(|mu_el| if *mu_el >= self.code.n(dind - 1) {
                    *mu_el -= self.code.n(dind - 1);
                });
                topcode = &code;
            } else {
                rescode &= self.code.G0code().is_code_element(&code);
            }

            let respoly = self.coeffring().eq_el(&self.coeffring().add(
                unipolyring.evaluate(poly, &self.coeffring().zero(), self.coeffring().identity()),
                unipolyring.evaluate(poly, &self.coeffring().one(), self.coeffring().identity())
            ), &tmp);
            tmp = unipolyring.evaluate(poly, &chall, self.coeffring().identity());

            challvec.push(chall);
            rescode && respoly
        }) && ({
            // challenges are sampled in order r_{d-1} -> r_0
            let challvec = challvec.into_iter().rev().collect_vec();

            if self.code.k(0) == 1 {
                let eqeval = MultilinearBasis::new(self.coeffring(), &z).evaluate(&challvec);
                // TODO: avoid div by zero
                let tmpenc = self.code.encode(&[self.coeffring().div(&tmp, &eqeval)]);
                (0..self.code.c()).all(|i| self.coeffring().eq_el(
                    &tmpenc[i],
                    &proof.code_els[d - 1][i]
                ))
            } else {
                let encm = self.code.encode(&proof.sumcheck_last);
                let kappa = self.code.k(0).ilog2() as usize;
                let mut eqevals = MultilinearBasisEvals::new(self.coeffring(),
                    &z[..kappa]).collect_vec();
                evalscalars_to_coeffscalars(self.coeffring(), kappa, &mut eqevals);
                let eval = sum_over_hypercube_withscalars(self.coeffring(),
                    eqevals.iter(), proof.sumcheck_last.iter());

                (0..self.code.n(0)).all(|i| {
                    self.coeffring().eq_el(&encm[i], &proof.code_els[d - 1][i])
                })
                &&
                self.coeffring().eq_el(&tmp, &self.coeffring().mul(eval,
                    MultilinearBasis::new(self.coeffring(), &z[kappa..]).evaluate(&challvec)))
            }
        })
    }

    // TODO: polycoeff shouldn't be an input when polyeval.is_some() since it isn't used in that
    // case? but keep pcs general for any pcs
    fn eval(&self, com: &BaseFoldCommitment<R>, z: &[El<R>], y: El<R>,
        polycoeff: Option<&[El<R>]>, polyeval: Option<&[El<R>]>) -> BaseFoldProof<R>
    {
        assert!(polycoeff.is_some() || polyeval.is_some()); // one is needed for doing sumcheck
        let d = self.code.d();
        let vc = self.polyring.indeterminate_count();
        let f = self.field();
        let unipolyring = self.get_unipolyring();
        
        let fsclone = self.fs.borrow().clone();

        // NOTE: should actually be faster to compute evals here and continue with evals
        // but not necessarily in blind setting
        let isevals = polyeval.is_some();
        let polyrepr = if let Some(unwr) = polyeval { unwr } else { polycoeff.unwrap() };

        let mut polys: Vec<El<DensePolyRing<R>>> = Vec::with_capacity(d);
        let bsc = BaseFoldSumcheckSpaceEfficient::new(&unipolyring, polyrepr, isevals, z);
        let mut hunivar = bsc.compute_round(&[], Some(f.clone_el(&y)));
        // let mut hunivar = bsc.compute_round(&[], None);
        polys.push(hunivar);

        let mut proofcodes: Vec<Vec<El<R>>> = Vec::with_capacity(d);
        let mut topcode = &com.code_el;

        let mut last: Vec<El<R>> = Vec::with_capacity(
            if self.code.k(0) == 1 {0} else {self.code.k(0)});

        let mut challvec: Vec<El<R>> = Vec::with_capacity(d - 1);
        
        for dind in (0..d).rev() {
            let chall = self.get_challenge();

            proofcodes.push({
                let (l, r) = topcode.split_at(self.code.n(dind));
                izip!(self.code.t(dind), l, r).map(|(ti, li, ri)|
                    interpdeg1(f, ti, li, ri, &chall)
                ).collect()
            });
            topcode = &proofcodes[d - 1 - dind];

            challvec.insert(0, chall);
            if dind != 0 {
                let tmpsum = unipolyring.evaluate(&polys[d - 1 - dind], &challvec[0], self.coeffring().identity());
                hunivar = bsc.compute_round(&challvec, Some(tmpsum));
                // hunivar = bsc.compute_round(&challvec, None);
                polys.push(hunivar);
            }
        }
    
        assert!(self.code.k(0).ilog2() as usize == vc - d);
        if self.code.k(0) > 1 {
            last = if isevals {
                let mut tmp = evaluate_at_fromevals(&f, vc, &challvec, polyrepr);
                // TODO: should I do this better?
                evals_to_coeffs_inplace(&f, vc - d, &mut tmp);
                tmp
            } else {
                evaluate_at_fromcoeff(&f, vc, &challvec, polyrepr)
            }
        }

        self.fs.replace(fsclone);

        BaseFoldProof {
            code_els: proofcodes,
            sumcheck_els: polys,
            sumcheck_last: last
        }
    }
}


// outputs f(a) where f(X) = interpolate((x, y1), (-x, y2))
// (could possibly use feanor_math::algorithms::interpolate)
pub fn interpdeg1<F>(field: &F, x: &El<F>, y1: &El<F>, y2: &El<F>, a: &El<F>) -> El<F>
    where F: RingStore<Type: Field>
{
    let mut tmpleft = field.add_ref(a, x);
    let mut tmpright = field.sub_ref(x, a);
    field.mul_assign_ref(&mut tmpleft, y1);
    field.mul_assign_ref(&mut tmpright, y2);
    field.div(
        &field.add(tmpleft, tmpright),
        &field.get_ring().mul_int_ref(x, 2)
    )
}


struct BaseFoldSumcheckSpaceEfficient<'a, R>
    where R: RingStore
{
    field: &'a R,
    vc: usize,
    unipolyring: &'a DensePolyRing<R>,
    polyrepr: &'a [El<R>],
    isevals: bool,
    z: &'a [El<R>]
}

impl<'a, F> BaseFoldSumcheckSpaceEfficient<'a, F>
    where F: RingStore<Type: Field>
{
    fn new(unipolyring: &'a DensePolyRing<F>, polyrepr: &'a [El<F>], isevals: bool, z: &'a [El<F>])
        -> Self
    {
        let vc = z.len();
        assert!(1 << vc == polyrepr.len());
        let field = unipolyring.get_ring().base_ring();
        Self { field, vc, unipolyring, polyrepr, isevals, z }
    }

    // assumes that polyrepr are either coeffs or evals belong to multilinear polynomial
    // O(N) time, O(logN) space if isevals == true (NOTE: called logN times from evals)
    // O(NlogN) time, O(logN) space if isevals == false
    // assumes polynomial is of the basefold form
    fn compute_round(&self, challenges: &[El<F>], sum: Option<El<F>>) -> El<DensePolyRing<F>> {

        let i = challenges.len();
        let vcmini = self.vc - i;

        let mut scalars = basefold_evalscalars(self.field, self.z, challenges, vcmini).collect_vec();
        if !self.isevals { evalscalars_to_coeffscalars(self.field, self.vc - 1, &mut scalars); }

        let iter1 = SCMultilinearIterator::new(self.polyrepr, vcmini, i, false);
        let tmp1 = sum_over_hypercube_withscalars(self.field, scalars.iter(), iter1);

        let zvcmini = &self.z[vcmini - 1];
        let tmp0 = if let Some(sum) = sum {
            // this is fhe-friendly
            let t = self.field.sub_ref_fst(&sum, self.field.mul_ref(&tmp1, &zvcmini));
            if !self.isevals { t } else {
                self.field.div(&t, &self.field.sub_ref_snd(self.field.one(), &zvcmini)) }
        } else {
            let iter0 = SCMultilinearIterator::new(self.polyrepr, vcmini, i, true);
            sum_over_hypercube_withscalars(self.field, scalars.iter(), iter0)
        };

        let oneminz = self.field.sub_ref_snd(self.field.one(), zvcmini);
        let twozminone = self.field.sub(self.field.get_ring().mul_int_ref(zvcmini, 2),
            self.field.one());
        let threezmintwo = self.field.sub(self.field.get_ring().mul_int_ref(zvcmini, 3),
            self.field.int_hom().map(2));
        let tmpsub = self.field.sub_ref(&tmp1, &tmp0);
        let (screpr, sumrepr) = if !self.isevals {(&twozminone, &tmp1)}
            else {(&threezmintwo, &tmpsub)};
        self.unipolyring.from_terms([
            (self.field.mul_ref(&oneminz, &tmp0), 0),
            (self.field.add(self.field.mul_ref(screpr, &tmp0),
                self.field.mul_ref(&oneminz, &tmp1)), 1),
            (self.field.mul_ref(&twozminone, sumrepr), 2)
        ])
    }
}


// TODO: remove redundancy with time-efficient? only used in space-efficient algorithm
pub fn basefold_evalscalars<'a, F>(field: &'a F, z: &'a [El<F>], challenges: &'a [El<F>],
    vcmini: usize) -> impl Iterator<Item = El<F>>
    where F: RingStore<Type: Field>
{
    let vc = z.len();
    let eq0 = MultilinearBasis::new(field, &z[vcmini..]).evaluate(challenges);
    let mut eq1 = MultilinearBasisEvals::new(field, challenges);
    let mut eq2 = MultilinearBasisEvals::new(field, &z[..(vcmini - 1)]);
    (0..(1 << (vc - 1))).map(move |i| {
        let tmp = field.mul_ref_fst(&eq0,
            field.mul_ref_fst(&eq1.cur, eq2.next().unwrap()));
        if (i + 1) % (1 << (vcmini - 1)) == 0 {
            eq1.next();
            eq2.reset();
        }
        tmp
    })
}


impl<'a, C, R> BaseFoldPCS<'a, R, C>
    where R: RingStore<Type: Field + FiniteRing> + Clone, C: FoldableCode<R = R>
{
    // TODO: this is super redundant!
    fn eval_te(&self, com: &BaseFoldCommitment<R>, z: &[El<R>], y: El<R>,
         polyeval: &[El<R>]) -> BaseFoldProof<R>
    {
        let d = self.code.d();
        let vc = self.polyring.indeterminate_count();
        let f = self.field();
        let unipolyring = self.get_unipolyring();
        
        let fsclone = self.fs.borrow().clone();

        let cfrome_deg2 = |pv: Vec<&El<R>>| {
            let c0 = f.clone_el(pv[0]);
            let c2 = f.div(&f.sub(f.add_ref(pv[1], pv[2]),
                f.get_ring().mul_int_ref(&c0, 2)), &f.int_hom().map(2));
            let c1 = f.sub_ref_snd(f.sub_ref(pv[1], &c0), &c2);
            unipolyring.from_terms([(c0, 0), (c1, 1), (c2, 2)])
        };

        let mut polys: Vec<El<DensePolyRing<R>>> = Vec::with_capacity(d);
        let bsc = BaseFoldSumcheckTimeEfficient::new(&unipolyring, polyeval, z);
        let mut hunivar = {
            let pe = bsc.compute_round(&[], Some(f.clone_el(&y)));
            // let pe = bsc.compute_round(&[], None);
            cfrome_deg2(pe.get_evals().collect_vec())
        };
        polys.push(hunivar);

        let mut proofcodes: Vec<Vec<El<R>>> = Vec::with_capacity(d);
        let mut topcode = &com.code_el;

        let mut last: Vec<El<R>> = Vec::with_capacity(
            if self.code.k(0) == 1 {0} else {self.code.k(0)});

        let mut challvec: Vec<El<R>> = Vec::with_capacity(d - 1);
        
        for dind in (0..d).rev() {
            let chall = self.get_challenge();

            proofcodes.push({
                let (l, r) = topcode.split_at(self.code.n(dind));
                izip!(self.code.t(dind), l, r).map(|(ti, li, ri)|
                    interpdeg1(f, ti, li, ri, &chall)
                ).collect()
            });
            topcode = &proofcodes[d - 1 - dind];

            challvec.insert(0, chall);
            if dind != 0 {
                let tmpsum = unipolyring.evaluate(&polys[d - 1 - dind], &challvec[0],
                    self.coeffring().identity());

                let mut hunivar = {
                    let pe = bsc.compute_round(&challvec, Some(tmpsum));
                    // let pe = bsc.compute_round(&[], None);
                    cfrome_deg2(pe.get_evals().collect_vec())
                };
                polys.push(hunivar);
            }
        }
    
        assert!(self.code.k(0).ilog2() as usize == vc - d);
        if self.code.k(0) > 1 {
            last = evaluate_at_fromevals(&f, vc, &challvec, polyeval);
            // TODO: should I do this better?
            evals_to_coeffs_inplace(&f, vc - d, &mut last);
        }

        self.fs.replace(fsclone);

        BaseFoldProof {
            code_els: proofcodes,
            sumcheck_els: polys,
            sumcheck_last: last
        }
    }
}


struct BaseFoldSumcheckTimeEfficient<'a, R>
    where R: RingStore
{
    field: &'a R,
    vc: usize,
    ws: RefCell<Vec<El<R>>>,
    z: &'a [El<R>],
    scalarstate: RefCell<El<R>>
}

impl<'a, F> BaseFoldSumcheckTimeEfficient<'a, F>
    where F: RingStore
{
    fn new(unipolyring: &'a DensePolyRing<F>, evals: &'a [El<F>], z: &'a [El<F>])
        -> Self
    {
        let vc = z.len();
        assert!(1 << vc == evals.len());
        let field = unipolyring.get_ring().base_ring();
        let ws = RefCell::new(evals.iter().map(|el| field.clone_el(el)).collect_vec());
        Self { field, vc, ws, z, scalarstate: RefCell::new(field.one()) }
    }
}

impl<'a, F> SumcheckBase<2> for BaseFoldSumcheckTimeEfficient<'a, F>
    where F: RingStore<Type: Field + FiniteRing>
{
    type F = F;

    fn field(&self) -> &F {
        self.field
    }

    fn varcount(&self) -> usize {
        self.vc
    }

    fn get_challenge(&self) -> El<F> {
        // we do not intend to call execute, so not needed
        unimplemented!()
    }

    fn get_other_eval_points(&self) -> [i32; 1] {
        [-1]
    }

    fn get_scalars<'b, 'c>(&'c self, challs: &'b [El<F>])
        -> impl Iterator<Item = (El<F>, El<F>)> + 'b
        where 'c: 'b
    {
        let field = self.field();
        let vcmini = self.vc - challs.len();
        let mut scstmut = self.scalarstate.borrow_mut();
        if vcmini < self.vc {
            field.mul_assign(&mut scstmut,
                MultilinearBasis::new(field, &self.z[vcmini..(vcmini + 1)]).evaluate(&challs[..1]));
        }
        MultilinearBasisEvals::new(field, &self.z[..(vcmini - 1)]).map(move |eqel| {
            let tmp = field.mul_ref_snd(eqel, &scstmut);
            let ztmp = &self.z[vcmini - 1];
            (
                field.mul_ref_fst(&tmp, field.sub_ref_snd(field.one(), ztmp)),
                field.mul_ref(&tmp, ztmp)
            )
        })
    }
}


impl<'a, F> Sumcheck<2, 1> for BaseFoldSumcheckTimeEfficient<'a, F>
    where F: RingStore<Type: Field + FiniteRing>
{
    type SCB = Self;

    fn get_base(&self) -> &Self {
        &self
    }

    fn get_workspace(&self) -> [&RefCell<Vec<El<F>>>; 1] {
        [&self.ws]
    }

    fn compute_term(ring: &F, at: [&El<F>; 1], scalar: &El<F>) -> El<F> {
        ring.mul_ref(scalar, &at[0])
    }

    fn check_eval(self, rX: Vec<El<F>>) -> bool {
        // handled concurrently in the Basefold sumcheck
        unimplemented!()
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use feanor_math::assert_el_eq;
    use feanor_math::rings::zn::ZnRingStore;
    use feanor_math::seq::VectorFn;
    use feanor_math::rings::zn::zn_64::Zn;
    use feanor_math::rings::finite::FiniteRingStore;
    use feanor_math::rings::field::AsField;

    use crate::util::gen_vector;
    use crate::multilinear::{from_hypercube_coeffs, sum_over_hypercube, coeffs_to_evals_inplace};

    const VREP: usize = 100;

    #[test]
    fn test_basefolding() {

        let field = Zn::new(65537).as_field().ok().unwrap();
        pub type FieldImpl = AsField<Zn>;

        let N = 7;
        let k0 = 1;
        let c = 2;
        let bf = BaseFoldPCS::new(&field, N, k0, c, VREP);
        let vc = bf.polyring().indeterminate_count();

        let randomcoeffs = gen_vector::<El<FieldImpl>>(||
            field.random_element(rand::random::<u64>), 1 << vc);
        let poly = from_hypercube_coeffs(bf.polyring(), &randomcoeffs);

        let topcode = bf.commit(&randomcoeffs).code_el;

        let chall = field.random_element(rand::random::<u64>);

        let d = bf.code.d();
        let foldedcode = bf.code.t(d-1).enumerate().map(|(i, ti)| {
            interpdeg1(&field, ti, &topcode[i], &topcode[i + bf.code.n(d-1)], &chall)
        }).collect_vec();

        let challconst = bf.polyring().create_term(field.clone_el(&chall),
            bf.polyring().create_monomial((0..vc).map(|_| 0)));
        let foldedevals = get_hypercube_coeffs(bf.polyring(),
            &bf.polyring().specialize(&poly, vc - 1, &challconst), vc - 1);
    
        assert!(foldedcode.len() == bf.code.n(d-1));
        assert!(foldedevals.len() == bf.code.k(d-1));

        let foldedcode2 = bf.code.encode(&foldedevals);
        assert!(foldedcode2.len() == bf.code.n(d-1));

        assert!((0..bf.code.n(d-1)).all(|i| field.eq_el(&foldedcode[i], &foldedcode2[i])));
    }

    #[test]
    fn test_basefoldpcs_slow() {
        
        let field = Zn::new(65537).as_field().ok().unwrap();
        pub type FieldImpl = AsField<Zn>;

        let N = 5;
        let k0 = 2;
        let c = 2;
        let bf = BaseFoldPCS::new(&field, N, k0, c, VREP);

        let randomcoeffs = gen_vector::<El<FieldImpl>>(||
            field.random_element(rand::random::<u64>), 1 << N);
        let poly = from_hypercube_coeffs(bf.polyring(), &randomcoeffs);

        let zinner = gen_vector::<El<FieldImpl>>(|| field.random_element(rand::random::<u64>), N);
        let z = (0..N).map_fn(|i| field.clone_el(&zinner[i]));
        let y = bf.polyring().evaluate(&poly, &z, bf.coeffring().identity());

        let com = bf.commit(&randomcoeffs);

        let zvec: Vec<_> = z.into_iter().collect();
        let proof = bf.eval_slow(&com, &zvec, field.clone_el(&y), &poly);

        assert!(bf.verify(com, &zvec, y, &randomcoeffs, proof))
    }

    #[test]
    fn test_basefoldsumchecksum() {

        let field = Zn::new(65537).as_field().ok().unwrap();
        pub type FieldImpl = AsField<Zn>;

        let N = 5;
        let polyring = MultivariatePolyRingImpl::new(field.clone(), N);

        let randomcoeffs = gen_vector::<El<FieldImpl>>(||
            field.random_element(rand::random::<u64>), 1 << N);
        let mut poly = from_hypercube_coeffs(&polyring, &randomcoeffs);
        
        // to make poly have variables of deg 2
        let zinner = gen_vector::<El<FieldImpl>>(|| field.random_element(rand::random::<u64>), N);
        let z = (0..N).map_fn(|i| field.clone_el(&zinner[i]));
        let zvec: Vec<_> = z.clone().into_iter().collect();
        let eq = MultilinearBasis::new(&field, &zvec).polynomial(&polyring);
        poly = polyring.mul_ref_fst(&poly, eq);
        
        let sum = sum_over_hypercube(&polyring, &poly, N, &[]);

        let unipolyring = DensePolyRing::new(field.clone(), "X");
        let mut hd = sumcheck_sum(&polyring, &unipolyring, &poly, N - 1);

        assert_el_eq!(field, &field.add(
            unipolyring.evaluate(&hd, &field.zero(), field.identity()),
            unipolyring.evaluate(&hd, &field.one(), field.identity())
        ), &sum);

        let bsc = BaseFoldSumcheckSpaceEfficient::new(&unipolyring, &randomcoeffs, false, &zvec);
        let mut hdfast = bsc.compute_round(&[], Some(sum));
        // let mut hdfast = bsc.compute_round(&[], None);
        
        assert_el_eq!(field, &field.add(
            unipolyring.evaluate(&hdfast, &field.zero(), field.identity()),
            unipolyring.evaluate(&hdfast, &field.one(), field.identity())
        ), &sum);
        assert_el_eq!(unipolyring, hd, hdfast);

        let mut rvec: Vec<El<FieldImpl>> = vec![];
        for ind in 1..=N-1 {
            //println!("tested: {}", ind - 1);
            let r = field.random_element(rand::random::<u64>);
            poly = polyring.specialize(&poly, N - ind,
                &polyring.create_term(field.clone_el(&r), polyring.create_monomial((0..N).map(|_| 0))));
            let sum = unipolyring.evaluate(&hd, &r, field.identity());
            hd = sumcheck_sum(&polyring, &unipolyring, &poly, N - ind - 1);
            assert_el_eq!(field, &field.add(
                unipolyring.evaluate(&hd, &field.zero(), field.identity()),
                unipolyring.evaluate(&hd, &field.one(), field.identity())
            ), &sum);
            rvec.insert(0, r);
            hdfast = bsc.compute_round(&rvec, Some(sum));
            // hdfast = bsc.compute_round(&rvec, None);
            assert_el_eq!(unipolyring, hd, hdfast);
        }
    }

    use tracing_subscriber::prelude::*;

    #[test]
    fn test_basefoldpcs_fast() {

        let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();

        let field = Zn::new(65537).as_field().ok().unwrap();
        pub type FieldImpl = AsField<Zn>;

        let N = 14;
        let k0 = 2;
        let c = 2;
        let bf = BaseFoldPCS::new(&field, N, k0, c, VREP);

        let randomcoeffs = gen_vector::<El<FieldImpl>>(||
            field.random_element(rand::random::<u64>), 1 << N);
        let poly = from_hypercube_coeffs(bf.polyring(), &randomcoeffs);

        let zinner = gen_vector::<El<FieldImpl>>(|| field.random_element(rand::random::<u64>), N);
        let z = (0..N).map_fn(|i| field.clone_el(&zinner[i]));
        let y = bf.polyring().evaluate(&poly, &z, bf.coeffring().identity());

        let com = bf.commit(&randomcoeffs);

        let zvec: Vec<_> = z.into_iter().collect();
        // let proof1 = bf.eval(&com, &zvec, y, Some(&randomcoeffs), None);

        let mut evals = randomcoeffs.iter().map(|el| field.clone_el(el)).collect_vec();
        coeffs_to_evals_inplace(&field, N, &mut evals);
        let proof2 = bf.eval(&com, &zvec, y, Some(&randomcoeffs), Some(&evals));
        // let proof2 = bf.eval_te(&com, &zvec, y, &evals);

        // assert!(bf.verify(com, &zvec, y, &randomcoeffs, proof1));
        assert!(bf.verify(com, &zvec, y, &randomcoeffs, proof2));
    }

}

