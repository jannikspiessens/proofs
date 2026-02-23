use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use itertools::Itertools;

use feanor_math::seq::VectorFn;
use feanor_math::delegate::{DelegateRing, DelegateRingImplFiniteRing};
use feanor_math::homomorphism::{Homomorphism, CanHomFrom, Identity, CanIsoFromTo, CanHom};
use feanor_math::integer::{BigIntRing, BigIntRingBase};
use feanor_math::ring::{ RingStore, RingBase, RingValue, RingExtension, El};
use feanor_math::rings::{
    extension::{
        extension_impl::FreeAlgebraImpl,
        FreeAlgebraStore
    },
    zn::{ ZnRing, ZnRingStore },
    finite::FiniteRing,
    direct_power::{DirectPowerRing, DirectPowerRingBase},
    field::{AsField, AsFieldBase}
};
use feanor_math::divisibility::DivisibilityRing;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::algorithms::{
    fft::{FFTAlgorithm, cooley_tuckey::CooleyTuckeyFFT},
    unity_root::get_prim_root_of_unity
};

use crate::{
    FSRng,
    lattice::gen_vector_infbnd,
    util::matmul::{MatrixMul, DenseMatrixMul},
};


pub const ZZbig: BigIntRing = BigIntRing::RING;


// NOTE *: only zn_64 and zn_big use the impl_field_wrap_unwrap_homs! to implement the
// AsFieldBase<R>: CanHomFrom<R::Type> homomorphism
// TODO: implement this generically for all R where R::Type: DivisibilityRing in feanor_math?
pub struct ABDLOPRingBase<R, const N: usize>
    where R: RingStore<Type: DivisibilityRing>, AsFieldBase<R>: CanHomFrom<R::Type> // NOTE: see *
{
    pr: DirectPowerRing<AsField<R>, N>,
    fft: CooleyTuckeyFFT<AsFieldBase<R>, AsFieldBase<R>, Identity<AsField<R>>>,
    hom: CanHom<R, AsField<R>>
}

impl<R, const N: usize> PartialEq for ABDLOPRingBase<R, N>
    where R: RingStore<Type: DivisibilityRing>, AsFieldBase<R>: CanHomFrom<R::Type>
{
    fn eq(&self, other: &Self) -> bool {
        self.fft.eq(&other.fft) && self.pr.get_ring().eq(other.pr.get_ring())
    }
}

impl<R, const N: usize> DelegateRing for ABDLOPRingBase<R, N>
    where R: RingStore<Type: DivisibilityRing>, AsFieldBase<R>: CanHomFrom<R::Type>
{
    // TODO: basering of DirectPowerRing does not actually have to be field,
    // (for some reason this is required for the get_prim_root_of_unity method)
    type Base = DirectPowerRingBase<AsField<R>, N>;
    // NOTE: ideally implement like below and then explicitly implement RingBase
    // similar to fheanor ManagedDoubleRNSRingBase
    // the bool remembers whether it is in NTT form or not
    // type Element = (El<DirectPowerRing<RFFT, N>>, bool);
    type Element = El<DirectPowerRing<AsField<R>, N>>;
 
    fn get_delegate(&self) -> &Self::Base {
        self.pr.get_ring()
    }
 
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element
    { el }
 
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element)
        -> &'a mut <Self::Base as RingBase>::Element { el }
 
    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
}

impl<R, const N: usize> DelegateRingImplFiniteRing  for ABDLOPRingBase<R, N>
    where R: RingStore<Type: DivisibilityRing>, AsFieldBase<R>: CanHomFrom<R::Type>
{}

impl<R, const N: usize> ABDLOPRingBase<R, N>
    where R: RingStore<Type: ZnRing> + Clone, AsFieldBase<R>: CanHomFrom<R::Type>
{
    pub fn new(basering: R) -> Self {
        assert!(N.is_power_of_two());

        let field = basering.clone().as_field().ok().expect("Provided ring is not a field");
        let root = get_prim_root_of_unity(&field, N)
            .expect(format!("Field does not have {}th primitive root of unity", N).as_str());
        let fft = CooleyTuckeyFFT::new(field.clone(), root, N.ilog2() as usize);
        let pr = DirectPowerRing::new(field.clone());
        let hom = field.into_can_hom(basering).ok().unwrap();

        Self { pr, fft, hom }
    }
}

pub type ABDLOPRing<R, const N: usize> = RingValue<ABDLOPRingBase<R,N>>;


type ExtR<R> = FreeAlgebraImpl<AsField<R>,[El<AsField<R>>; 2]>;
type ExtFb<R> = AsFieldBase<ExtR<R>>;
type ExtF<R> = RingValue<ExtFb<R>>;

pub struct ABDLOPRingExtBase<R, const N: usize>
    where R: RingStore<Type: DivisibilityRing + CanIsoFromTo<R::Type>>,
          AsFieldBase<R>: CanHomFrom<R::Type>
{
    pr: DirectPowerRing<ExtF<R>, N>,
    fft: CooleyTuckeyFFT<ExtFb<R>, ExtFb<R>, Identity<ExtF<R>>>,
    basehom: CanHom<R, AsField<R>>,
    hom: CanHom<ExtR<R>, ExtF<R>>
}

impl<R, const N: usize> PartialEq for ABDLOPRingExtBase<R, N>
    where R: RingStore<Type: DivisibilityRing + CanIsoFromTo<R::Type>>,
          AsFieldBase<R>: CanHomFrom<R::Type>
{
    fn eq(&self, other: &Self) -> bool {
        self.fft.eq(&other.fft) && self.pr.get_ring().eq(other.pr.get_ring())
    }
}

impl<R, const N: usize> DelegateRing for ABDLOPRingExtBase<R, N>
    where R: RingStore<Type: DivisibilityRing + CanIsoFromTo<R::Type>>,
          AsFieldBase<R>: CanHomFrom<R::Type>
{
    type Base = DirectPowerRingBase<ExtF<R>, N>;
    type Element = El<DirectPowerRing<ExtF<R>, N>>;
 
    fn get_delegate(&self) -> &Self::Base {
        self.pr.get_ring()
    }
 
    fn delegate_ref<'a>(&self, el: &'a Self::Element) -> &'a <Self::Base as RingBase>::Element
    { el }
 
    fn delegate_mut<'a>(&self, el: &'a mut Self::Element)
        -> &'a mut <Self::Base as RingBase>::Element { el }
 
    fn delegate(&self, el: Self::Element) -> <Self::Base as RingBase>::Element { el }

    fn rev_delegate(&self, el: <Self::Base as RingBase>::Element) -> Self::Element { el }
}

impl<R, const N: usize> DelegateRingImplFiniteRing  for ABDLOPRingExtBase<R, N>
    where R: RingStore<Type: DivisibilityRing + CanIsoFromTo<R::Type>>,
          AsFieldBase<R>: CanHomFrom<R::Type>
{}

pub type ABDLOPRingExt<R, const N: usize> = RingValue<ABDLOPRingExtBase<R,N>>;


impl<R, const N: usize> ABDLOPRingExtBase<R, N>
    where R: RingStore<Type: DivisibilityRing + ZnRing + CanIsoFromTo<R::Type>> + Clone,
          AsFieldBase<R>: CanHomFrom<R::Type>
{
    pub fn new(basering: R) -> Self {
        assert!(N.is_power_of_two());

        let field = basering.clone().as_field().ok()
            .expect("Provided ring is not a field");

        let basehom = field.clone().into_can_hom(basering).ok().unwrap();
        let fring = FreeAlgebraImpl::new(field.clone(), 2, [field.neg_one(), field.zero()]);
        let gf = fring.as_field().ok().unwrap();

        // TODO: clean this
        let fring = FreeAlgebraImpl::new(field.clone(), 2, [field.neg_one(), field.zero()]);
        let gfclone = fring.as_field().ok().unwrap();
        let fring = FreeAlgebraImpl::new(field.clone(), 2, [field.neg_one(), field.zero()]);
        let hom = gfclone.into_can_hom(fring).ok().unwrap();

        // TODO: clean this
        let fring = FreeAlgebraImpl::new(field.clone(), 2, [field.neg_one(), field.zero()]);
        let gfclone = fring.as_field().ok().unwrap();

        let root = get_prim_root_of_unity(&gf, N)
            .expect("Fp2 does not have primitive root of unity");

        let fft = CooleyTuckeyFFT::new(gfclone, root, N.ilog2() as usize);
        let pr = DirectPowerRing::<_, N>::new(gf);

        Self { pr, fft, basehom, hom }
    }

    fn fring(&self) -> &ExtR<R> { &self.hom.domain() }
}


pub trait ABDLOPRingNTT<const N: usize>: RingStore<Type: FiniteRing> {

    type BaseRing: RingStore<Type: CanHomFrom<BigIntRingBase> + ZnRing>;
    type FFTRing: RingStore<Type: DivisibilityRing>;

    fn base_ring(&self) -> &Self::BaseRing;

    fn FFTring(&self) -> &Self::FFTRing;
    
    fn to_FFTRing(&self, inp: El<Self::BaseRing>) -> El<Self::FFTRing>;
    fn to_BaseRing(&self, inp: El<Self::FFTRing>) -> El<Self::BaseRing>;

    fn ffter(&self) -> &CooleyTuckeyFFT<<Self::FFTRing as RingStore>::Type,
        <Self::FFTRing as RingStore>::Type, Identity<Self::FFTRing>>;

    // NOTE: weird that these methods are needed
    fn to_array(inp: El<Self>) -> El<DirectPowerRing<Self::FFTRing, N>>;
    fn to_array_mut(inp: &mut El<Self>) -> &mut El<DirectPowerRing<Self::FFTRing, N>>;
    fn from_array(inp: El<DirectPowerRing<Self::FFTRing, N>>) -> El<Self>;

    fn ntt(&self, el: &mut El<Self>) {
        self.ffter().fft(Self::to_array_mut(el), self.FFTring())
    }

    fn intt(&self, el: &mut El<Self>) {
        self.ffter().inv_fft(Self::to_array_mut(el), self.FFTring())
    }

    fn scalar_mul_ref(&self, scalar: &El<Self::BaseRing>, inp: &El<Self>) -> El<Self> {
        self.scalar_mul(scalar, self.clone_el(inp))
    }

    fn scalar_mul(&self, scalar: &El<Self::BaseRing>, inp: El<Self>) -> El<Self> {
        Self::from_array(Self::to_array(inp).map(|el|
            self.FFTring().mul(el, self.to_FFTRing(self.base_ring().clone_el(scalar)))))
    }
}

impl<R, const N: usize> ABDLOPRingNTT<N> for ABDLOPRing<R, N>
    where R: RingStore<Type: ZnRing + CanHomFrom<BigIntRingBase>>,
          AsFieldBase<R>: CanHomFrom<R::Type>
{
    type BaseRing = R;
    type FFTRing = AsField<R>;

    fn base_ring(&self) -> &Self::BaseRing { self.get_ring().hom.domain() }

    fn FFTring(&self) -> &Self::FFTRing { self.get_ring().base_ring() }

    fn to_FFTRing(&self, inp: El<Self::BaseRing>) -> El<Self::FFTRing> {
        self.get_ring().hom.map(inp)
    }

    fn to_BaseRing(&self, inp: El<Self::FFTRing>) -> El<Self::BaseRing> {
        self.FFTring().get_ring().unwrap_element(inp)
    }

    fn ffter(&self) -> &CooleyTuckeyFFT<<Self::FFTRing as RingStore>::Type, <Self::FFTRing as RingStore>::Type, Identity<Self::FFTRing>>
    { &self.get_ring().fft }

    fn to_array(inp: El<Self>) -> El<DirectPowerRing<Self::FFTRing, N>> { inp }
    fn to_array_mut(inp: &mut El<Self>) -> &mut El<DirectPowerRing<Self::FFTRing, N>> { inp }
    fn from_array(inp: El<DirectPowerRing<Self::FFTRing, N>>) -> El<Self> { inp }
}

impl<R, const N: usize> ABDLOPRingNTT<N> for ABDLOPRingExt<R, N>
    where R: RingStore<Type: ZnRing + CanIsoFromTo<R::Type>
        + CanHomFrom<BigIntRingBase>> + Clone, AsFieldBase<R>: CanHomFrom<R::Type>
{
    type BaseRing = R;
    type FFTRing = ExtF<R>;

    fn base_ring(&self) -> &Self::BaseRing { &self.get_ring().basehom.domain() }

    fn FFTring(&self) -> &Self::FFTRing { self.get_ring().base_ring() }

    fn to_FFTRing(&self, inp: El<Self::BaseRing>) -> El<Self::FFTRing> {
        let ring = self.get_ring();
        let basering = ring.base_ring().get_ring().base_ring();
        ring.hom.map(ring.fring().from_canonical_basis([ring.basehom.map(inp), basering.zero()]))
    }

    fn to_BaseRing(&self, inp: El<Self::FFTRing>) -> El<Self::BaseRing> {
        let ring = self.get_ring();
        ring.basehom.codomain().get_ring().unwrap_element(ring.fring().wrt_canonical_basis(
            &self.FFTring().get_ring().unwrap_element(inp)).at(0))
    }

    fn ffter(&self) -> &CooleyTuckeyFFT<<Self::FFTRing as RingStore>::Type,
        <Self::FFTRing as RingStore>::Type, Identity<Self::FFTRing>>
    { &self.get_ring().fft }

    fn to_array(inp: El<Self>) -> El<DirectPowerRing<Self::FFTRing, N>> { inp }
    fn to_array_mut(inp: &mut El<Self>) -> &mut El<DirectPowerRing<Self::FFTRing, N>> { inp }
    fn from_array(inp: El<DirectPowerRing<Self::FFTRing, N>>) -> El<Self> { inp }

    // TODO: make ntt methods go back to basering?
}


pub struct ABDLOPcommitment<R, const N: usize>
    where R: ABDLOPRingNTT<N>
{
    // NOTE: always assumed to be in NTT form
    t: Vec<El<R>>
}

impl<R, const N: usize> Deref for ABDLOPcommitment<R, N>
    where R: ABDLOPRingNTT<N>
{
    type Target = Vec<El<R>>;

    fn deref(&self) -> &Self::Target { &self.t }
}

impl<R, const N: usize> DerefMut for ABDLOPcommitment<R, N>
    where R: ABDLOPRingNTT<N>
{
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.t }
}


pub struct ABDLOPmessage<'a, R, const N: usize>
    where R: ABDLOPRingNTT<N>
{
    // NOTE: both always assumed to be in NTT form
    s1: Option<&'a Vec<El<R>>>,
    m: Option<&'a Vec<El<R>>>
}

impl<'a, R, const N: usize> ABDLOPmessage<'a, R, N>
    where R: ABDLOPRingNTT<N>
{
    pub fn new(s1: &'a Option<Vec<El<R>>>, m: &'a Option<Vec<El<R>>>) -> Self
    {
        assert!(s1.is_some() || m.is_some());
        Self { s1: s1.as_ref(), m: m.as_ref() }
    }

    pub fn s1(&self) -> Option<&'a Vec<El<R>>> { self.s1 }

    pub fn m(&self) -> Option<&'a Vec<El<R>>> { self.m }
}


pub struct ABDLOPopening<R, const N: usize>
    where R: ABDLOPRingNTT<N>
{
    // NOTE: always assumed to be in NTT form
    s2: Vec<El<R>>
}

impl<R, const N: usize> Deref for ABDLOPopening<R, N>
    where R: ABDLOPRingNTT<N>
{
    type Target = Vec<El<R>>;

    fn deref(&self) -> &Self::Target { &self.s2 }
}


#[derive(Clone, Copy)]
pub enum ABDLOPparts { Ajtai, BDLOP }


struct ABDLOPprecomp<R, const N: usize>
    where R: ABDLOPRingNTT<N>
{
    // NOTE: all always assumed to be in NTT form
    s2: RefCell<ABDLOPopening<R,N>>,
    A2s2: RefCell<Vec<El<R>>>,
    Bs2: Option<Vec<El<R>>>
}

impl<R, const N: usize> ABDLOPprecomp<R,N>
    where R: ABDLOPRingNTT<N>
{
    fn empty() -> Self {
        Self {
            s2: RefCell::new(ABDLOPopening{s2: Vec::new() }),
            A2s2: RefCell::new(Vec::new()),
            Bs2: None
        }
    }

    fn is_some(&self) -> bool {
        self.Bs2.is_some()
    }

    fn get_ref(&self) -> (ABDLOPopening<R,N>, Vec<El<R>>, &Option<Vec<El<R>>>)
    {
        assert!(self.is_some());
        let s2 = self.s2.replace(ABDLOPopening{s2: Vec::new() });
        let A2s2 = self.A2s2.replace(Vec::new());
        (s2, A2s2, &self.Bs2)
    }
}


pub struct ABDLOP<'a, R, const N: usize>
    where R: ABDLOPRingNTT<N>
{
    ring: &'a R,
    // TODO: is there is noticable difference in performance when using RefCell for the RNG?
    rng: RefCell<FSRng>,
    bnd1: Option<El<BigIntRing>>, // TODO: add possibility for 2norm bounds
    bnd2: El<BigIntRing>,
    A1: Option<DenseMatrixMul<'a, R>>, // NOTE: matrix elements stored in NTT form
    A2: DenseMatrixMul<'a, R>,
    B: Option<DenseMatrixMul<'a, R>>,
    precomp: RefCell<ABDLOPprecomp<R,N>>
}

impl<'a, R, const N: usize> ABDLOP<'a, R, N>
    where R: ABDLOPRingNTT<N>
{
    pub fn random(ring: &'a R, mut rng: FSRng,
        n: usize, l: Option<usize>, m1: Option<usize>, m2: usize,
        bnd1: Option<El<BigIntRing>>, bnd2: El<BigIntRing>) -> Self
    {
        // TODO: doing way to much in the extension field, should just keep everything in basefield
        // and only switch to extension right before NTTs
        let A1 = if bnd1.is_none() { None }
            else { Some(DenseMatrixMul::random(ring, &mut rng, n, m1.unwrap(), "ABDLOP_A1")) };
        let A2 = DenseMatrixMul::random(ring, &mut rng, n, m2, "ABDLOP_A2");
        let B = if let Some(l) = l {
            Some(DenseMatrixMul::random(ring, &mut rng, l, m2, "ABDLOP_B")) } else { None };
        let precomp = RefCell::new(ABDLOPprecomp::empty());
        Self { ring, rng: RefCell::new(rng), bnd1, bnd2, A1, A2, B, precomp }
    }
}

impl<'a, R, const N: usize> ABDLOP<'a, R, N>
    where R: ABDLOPRingNTT<N>
{
    pub fn ring(&self) -> &R { self.ring }

    pub fn base_ring(&self) -> &R::BaseRing { self.ring().base_ring() }

    pub fn rng(&self) -> &RefCell<FSRng> { &self.rng }

    pub fn has_ajtai(&self) -> bool { self.A1.is_some() }

    pub fn has_bdlop(&self) -> bool { self.B.is_some() }

    pub fn get_m1(&self) -> Option<usize> { self.A1.as_ref().map(|x| x.columns()) }

    pub fn get_m2(&self) -> usize { self.A2.columns() }

    pub fn get_bnd1(&self) -> Option<&El<BigIntRing>> { self.bnd1.as_ref() }

    pub fn get_bnd2(&self) -> &El<BigIntRing> { &self.bnd2 }

    pub fn get_A1(&self) -> Option<&DenseMatrixMul<'a, R>> { self.A1.as_ref() }

    pub fn get_A2(&self) -> &DenseMatrixMul<'a, R> { &self.A2 }

    pub fn get_B(&self) -> Option<&DenseMatrixMul<'a, R>> { self.B.as_ref() }

    pub fn comlen(&self) -> usize { self.A2.rows() + self.get_B().map_or(0, |x| x.rows()) }

    // TODO: make next four functions provided methods for ABDLOPRingNTT?
    fn to_ntt_ring_ref(&self, inp: &[El<<R as ABDLOPRingNTT<N>>::BaseRing>],
        prefixlen: Option<usize>) -> Vec<El<R>>
    {
        assert!(prefixlen.is_none_or(|x| x < N));
        let tmp = prefixlen.unwrap_or(0);
        self.to_ntt_ring(
            (0..tmp).map(|_| self.base_ring().zero())
            .chain(inp.iter().map(|el| self.base_ring().clone_el(el)))
            .chain(((dbg!(inp.len()) + dbg!(tmp))..dbg!((inp.len() + tmp).next_multiple_of(N)))
                .map(|_| self.base_ring().zero())).collect()
        )
    }

    // TODO: input iterator?
    pub fn to_ntt_ring(&self, inp: Vec<El<<R as ABDLOPRingNTT<N>>::BaseRing>>) -> Vec<El<R>>
    {
        assert!(inp.len() % N == 0);
        inp.into_iter().map(|el|
            self.ring().to_FFTRing(el)).collect_vec().into_chunks::<N>().into_iter().map(|el|
                R::from_array(el)).collect_vec()
    }

    pub fn to_base_ring_ref(&self, inp: &[El<R>]) -> Vec<El<<R as ABDLOPRingNTT<N>>::BaseRing>>
    {
        self.to_base_ring(inp.iter().map(|el| self.ring().clone_el(el)).collect())
    }

    // TODO: input iterator?
    pub fn to_base_ring(&self, inp: Vec<El<R>>) -> Vec<El<<R as ABDLOPRingNTT<N>>::BaseRing>>
    {
        inp.into_iter().flat_map(|el| R::to_array(el))
            .map(|el| self.ring().to_BaseRing(el)).collect()
    }

    pub fn gen_m(&self, inp: Vec<El<<R as ABDLOPRingNTT<N>>::BaseRing>>) -> Vec<El<R>> {
        if !(inp.len() % N == 0) {
            panic!("Input to ABDLOP::gen_m must have length divisible by N");
        }
        self.to_ntt_ring(inp)
    }

    pub fn gen_s1(&self, inp: Vec<El<<R as ABDLOPRingNTT<N>>::BaseRing>>) -> Vec<El<R>> {
        if !(inp.len() % N == 0) {
            panic!("Input to ABDLOP::gen_s1 must have length divisible by N");
        }
        let mut res = self.to_ntt_ring(inp);
        if !self.check_inf_norm::<false>(&res, self.bnd1.as_ref().unwrap()) {
            panic!("Input to ABDLOP::gen_s1 must be bounded by ABDLOP::get_bnd1");
        }
        res.iter_mut().for_each(|x| self.ring().ntt(x));
        res
    }

    // inputs assumed to be in NTT form
    fn commit_ajtai(&'a self, s1opt: Option<&'a Vec<El<R>>>, s2: &'a [El<R>])
        -> Box<dyn Iterator<Item = El<R>> + 'a>
    {
        let s2iter = self.A2.mulit(s2);
        if let Some(s1) = s1opt {
            Box::new(self.get_A1().unwrap().mulit(s1).zip(s2iter).map(|(l, r)| self.ring.add(l, r)))
        } else { Box::new(s2iter) }
    }

    fn commit_bdlop(&self, m: &[El<R>], s2: &[El<R>], offset: Option<usize>)
        -> impl Iterator<Item = El<R>>
    {
        assert!(self.has_bdlop());
        let B = self.get_B().unwrap();
        assert!(B.rows() >= m.len());
        let ofs = offset.unwrap_or(0);
        B.submatmul(ofs..(ofs+m.len()), 0..B.columns(), s2).zip(m)
            .map(|(l, r)| self.ring.add_ref_snd(l, r))
    }

    fn gen_s2(&self) -> Vec<El<R>> {
        let mut rngmut = self.rng.borrow_mut();
        let mut s2 = self.to_ntt_ring(gen_vector_infbnd(self.base_ring(),
            &mut rngmut, &self.bnd2, self.A2.columns()*N));
        s2.iter_mut().for_each(|s2el| self.ring().ntt(s2el)); // go to NTT representation
        s2
    }

    pub fn commit(&self, mes: &ABDLOPmessage<R,N>) -> (ABDLOPcommitment<R,N>, ABDLOPopening<R,N>)
    {
        assert!(!self.has_ajtai() || mes.s1.is_some());

        if self.precomp.borrow().is_some() {
            return self.commit_precomp(mes)
        } else {
            println!("ABDLOP: call precomp first for faster committing!");
        }

        let s2 = self.gen_s2();
        let mut t = Vec::with_capacity(self.comlen());
        t.extend(self.commit_ajtai(mes.s1(), &s2));
        if let Some(m) = &mes.m {
            t.extend(self.commit_bdlop(m, &s2, None));
        }
        (ABDLOPcommitment{ t }, ABDLOPopening{ s2 })
    }

    fn check_inf_norm<const INTT: bool>(&self, inp: &[El<R>], bnd: &El<BigIntRing>) -> bool {
        let intring = self.base_ring().integer_ring();
        let hom = intring.can_hom(&ZZbig).unwrap();
        let bnd = hom.map_ref(bnd);
       
        inp.into_iter().all(|el| {
            let mut tmp = self.ring().clone_el(el);
            if INTT {self.ring().intt(&mut tmp)};
            R::to_array(tmp).into_iter().all(|ell|
                intring.is_leq(&self.base_ring().smallest_lift(self.ring().to_BaseRing(ell)), &bnd)
            )
        })
    }

    pub fn open(&self, com: &ABDLOPcommitment<R,N>,
        mes: &ABDLOPmessage<R,N>, op: &ABDLOPopening<R,N>) -> bool
    {
        let c1 = mes.s1.as_ref().is_none_or(|x|
            self.check_inf_norm::<true>(x, self.bnd1.as_ref().unwrap())
        );
        let c2 = self.check_inf_norm::<true>(op, &self.bnd2);
        if !(c1 && c2 && com.len() == self.comlen()) { return false };

        let mut iter: Box<dyn Iterator<Item = El<R>>> = self.commit_ajtai(mes.s1(), op);
        if let Some(m) = &mes.m {
            iter = Box::new(iter.chain(self.commit_bdlop(m, op, None)));
        }

        iter.zip(com.iter()).all(|(l, r)| self.ring.eq_el(&l, r))
    }

    pub fn append_commit(&self, com: &mut ABDLOPcommitment<R,N>, op: &ABDLOPopening<R,N>,
        mb: &[El<<R as ABDLOPRingNTT<N>>::BaseRing>], actlen: Option<usize>)
    {
        let comlen = com.len();
        let n = self.A2.rows();
        assert!(comlen >= n);
        assert!(self.has_bdlop());

        assert!(actlen.is_none_or(|x| x < (comlen-n)*N && x > (comlen-n-1)*N ));
        assert!(self.comlen()*N >= n*N + mb.len() + actlen.unwrap_or(0));

        let m = self.to_ntt_ring_ref(mb, actlen.map(|x| x % N));
        println!("len m: {}", m.len());

        let offs = actlen.map_or(comlen - n, |x| x/N);
        let precomp = self.precomp.borrow();
        
        let mut newcom: Box<dyn Iterator<Item = El<R>>> = if precomp.is_some() {
            Box::new(precomp.Bs2.as_ref().unwrap()[offs..offs+m.len()].iter().zip(m)
                .map(|(l,r)| self.ring.add_ref_fst(l, r)))
        } else {
            println!("ABDLOP: call precomp first for faster committing!");
            Box::new(self.commit_bdlop(&m, op, Some(offs)))
        };

        if let Some(actlen) = actlen {
            let mut first = newcom.next().unwrap();
            R::to_array_mut(&mut first)[..(actlen % N)].iter_mut()
                .for_each(|el| *el = self.ring().FFTring().zero());
            self.ring().add_assign(&mut com[comlen-1], first);
        }

        com.extend(newcom);

        // NOTE: prevent partial leakage of Bs2 mask in the clear
        let totlen = actlen.unwrap_or(0) + mb.len();
        if totlen % N != 0 {
            let comlen = com.len();
            let tmp = &mut com[comlen-1];
            R::to_array_mut(tmp)[(totlen % N)..].iter_mut().for_each(|el|
                *el = self.ring().FFTring().zero());
        }
    }

    pub fn precomp(&self) {
        let s2 = ABDLOPopening{ s2: self.gen_s2() };
        let A2s2 = RefCell::new(self.A2.mul(&s2));
        let Bs2 = self.B.as_ref().map(|x| x.mul(&s2));
        self.precomp.replace(ABDLOPprecomp{ s2: RefCell::new(s2), A2s2, Bs2 });
    }

    fn commit_precomp(&self, mes: &ABDLOPmessage<R,N>)
        -> (ABDLOPcommitment<R,N>, ABDLOPopening<R,N>)
    {
        let precomp = self.precomp.borrow();
        if !precomp.is_some() {
            panic!("Call ABDLOP::precomp before calling ABDLOP::commit_precomp!");
        }
        let (s2, A2s2, Bs2) = precomp.get_ref();

        let mut t = Vec::with_capacity(self.comlen());
        if let Some(A1) = self.get_A1() {
            t.extend(A1.mulit(mes.s1().unwrap()).zip(A2s2.into_iter()).map(|(l, r)|
                self.ring.add(l, r)))
        } else { t.extend(A2s2) };

        if let Some(m) = mes.m {
            t.extend(Bs2.as_ref().unwrap().into_iter().zip(m).map(|(l,r)| self.ring.add_ref(l,r)))
        }
        (ABDLOPcommitment{ t }, s2)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use feanor_math::rings::zn::zn_big::Zn;
    use feanor_math::integer::IntegerRing;

    use crate::util::gen_random;

    #[test]
    fn test_abdlop_ring() {

        let ring = feanor_math::rings::zn::zn_64::Zn::new(65537);

        const N: usize = 1 << 12;
        let abdlopring = RingValue::from(ABDLOPRingBase::<_, N>::new(ring.clone()));

        let n = 2;
        let l = 2;
        let m2 = 1;

        let inthom = ZZbig.int_hom();
        let bnd2 = inthom.map(1 << 10);

        // let m1 = None;
        // let s1 = None;
        // let bnd1 = None;
        let m1 = Some(1);
        let bnd1 = Some(inthom.map(1 << 10));

        // let rng = rand::rng();
        let rng = <rand::rngs::StdRng as rand::SeedableRng>::from_os_rng();
        let abdlop = ABDLOP::random(&abdlopring, rng, n, Some(l), m1, m2, bnd1, bnd2);

        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::from_os_rng();
        let s1 = Some(abdlop.gen_s1(gen_vector_infbnd(&ring, &mut rng,
                abdlop.get_bnd1().as_ref().unwrap(), m2*N)));
        let m = Some(abdlop.gen_m(gen_random(&ring, &mut rng, l*N)));

        abdlop.precomp();

        let mes = ABDLOPmessage::new(&s1, &m);
        let now = std::time::SystemTime::now();
        let (com, op) = abdlop.commit(&mes);
        println!("TEST commit time: {}ms", now.elapsed().unwrap().as_millis());

        assert!(abdlop.open(&com, &mes, &op));
    }

    #[test]
    fn test_abdlop_append() {

        let ring = feanor_math::rings::zn::zn_64::Zn::new(65537);

        const N: usize = 1 << 12;
        let abdlopring = RingValue::from(ABDLOPRingBase::<_, N>::new(ring.clone()));

        let n = 2;
        let l = 3;
        let m2 = 1;

        let inthom = ZZbig.int_hom();
        let bnd2 = inthom.map(1 << 10);

        // let m1 = None;
        // let s1 = None;
        // let bnd1 = None;
        let m1 = Some(1);
        let bnd1 = Some(inthom.map(1 << 10));

        // let rng = rand::rng();
        let rng = <rand::rngs::StdRng as rand::SeedableRng>::from_os_rng();
        let abdlop = ABDLOP::random(&abdlopring, rng, n, Some(l), m1, m2, bnd1, bnd2);

        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::from_os_rng();
        let s1 = Some(abdlop.gen_s1(gen_vector_infbnd(&ring, &mut rng,
                abdlop.get_bnd1().as_ref().unwrap(), m2*N)));
        let mut m = gen_random(&ring, &mut rng, N);

        println!("Precomp");
        abdlop.precomp();

        println!("Commit");
        let mopt = Some(abdlop.gen_m(m.iter().map(|el| ring.clone_el(el)).collect()));
        let mes = ABDLOPmessage::new(&s1, &mopt);
        let (mut com, op) = abdlop.commit(&mes);

        println!("Append 1");
        let mext = gen_random(&ring, &mut rng, N - 15);
        abdlop.append_commit(&mut com, &op, &mext, None);

        println!("Append 2");
        let mext2 = gen_random(&ring, &mut rng, N + 10);
        abdlop.append_commit(&mut com, &op, &mext2, Some(2*N - 15));

        println!("Append 3");
        let mext3 = gen_random(&ring, &mut rng, 5);
        abdlop.append_commit(&mut com, &op, &mext3, Some(l*N - 5));

        m.extend(mext);
        m.extend(mext2);
        m.extend(mext3);
        let mext = Some(abdlop.gen_m(m));
        let mes = ABDLOPmessage::new(&s1, &mext);

        assert!(abdlop.open(&com, &mes, &op));
    }

    #[test]
    fn test_abdlop_ext() {

        let p = ZZbig.get_ring().parse("864175120484581453683482079962486176185193500155369104423588921177379322250834082489183304374038697487834084609675858746433355728113743766078731283595263", 10).unwrap();
        let ring = Zn::new(ZZbig, p);
        const N: usize = 1 << 12;
        let abdlopring = RingValue::from(ABDLOPRingExtBase::<_, N>::new(ring.clone()));

        let n = 2;
        let l = 2;
        let m2 = 1;

        let inthom = ZZbig.int_hom();
        let bnd2 = inthom.map(1 << 10);

        // let m1 = None;
        // let s1 = None;
        // let bnd1 = None;
        let m1 = Some(1);
        let bnd1 = Some(inthom.map(1 << 10));

        // let rng = rand::rng();
        let rng = <rand::rngs::StdRng as rand::SeedableRng>::from_os_rng();
        let abdlop = ABDLOP::random(&abdlopring, rng, n, Some(l), m1, m2, bnd1, bnd2);


        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::from_os_rng();
        let s1 = Some(abdlop.gen_s1(gen_vector_infbnd(&ring, &mut rng,
                abdlop.get_bnd1().as_ref().unwrap(), m2*N)));
        let m = Some(abdlop.gen_m(gen_random(&ring, &mut rng, l*N)));

        abdlop.precomp();

        let mes = ABDLOPmessage::new(&s1, &m);
        let now = std::time::SystemTime::now();
        let (com, op) = abdlop.commit(&mes);
        println!("TEST commit time: {}ms", now.elapsed().unwrap().as_millis());

        assert!(abdlop.open(&com, &mes, &op));
    }
}

