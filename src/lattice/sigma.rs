use rand::RngCore;
use std::cell::RefCell;
use itertools::{Itertools, izip};

use feanor_math::integer::{BigIntRing, IntegerRingStore};
use feanor_math::ring::{ RingStore, El };
use feanor_math::rings::{
    zn::{zn_big::Zn, ZnRingStore},
    finite::FiniteRingStore
};

use crate::{
    FSRng,
    commit::abdlop::{
        ZZbig, ABDLOP, ABDLOPcommitment, ABDLOPmessage, ABDLOPopening, ABDLOPparts, ABDLOPRingTrait
    },
    lattice::{gen_infbnd, gen_vector_dgauss, gen_vector_latrejsampl, norm2, RejSamplModes},
    util::{
        gen_random, FiatShamirSim,
        matmul::{MatrixMul, SparseMatrixMul, DenseMatrixMul},
    }
};


pub struct LatSigmaLinRel<R, MM1, MMm, const N: usize>
    where R: ABDLOPRingTrait<N>, MM1: MatrixMul<R = R::BaseRing>, MMm: MatrixMul<R = R::BaseRing>
{
    R1: Option<MM1>,
    Rm: Option<MMm>,
    u: Vec<El<R::BaseRing>>
}

impl<R, MM1, MMm, const N: usize> LatSigmaLinRel<R, MM1, MMm, N>
    where R: ABDLOPRingTrait<N>, MM1: MatrixMul<R = R::BaseRing>, MMm: MatrixMul<R = R::BaseRing>
{
    fn empty() -> Self {
        Self { R1: None, Rm: None, u: Vec::new() }
    }

    fn is_some(&self) -> bool {
        self.R1.is_some() || self.Rm.is_some()
    }

    fn rows(&self) -> usize {
        assert!(self.is_some());
        let res = self.u.len();
        assert!(self.R1.as_ref().is_none_or(|x| x.rows() == res));
        assert!(self.Rm.as_ref().is_none_or(|x| x.rows() == res));
        res
    }

    fn compute_h<'a>(&self, ring: &'a R, g: &El<R>, gammas: &[El<R::BaseRing>],
        mes: &ABDLOPmessage<R,N>)
        -> (El<R>, Option<DenseMatrixMul<'a, R>>, Option<DenseMatrixMul<'a, R>>)
    {
        assert!(self.rows() == gammas.len());

        let mut R1NTT = self.R1.as_ref().map(|x| Vec::<El<R>>::with_capacity(x.columns()/N));
        let mut RmNTT = self.Rm.as_ref().map(|x| Vec::<El<R>>::with_capacity(x.columns()/N + 1));

        let mut h = ring.clone_el(g);
        
        gammas.iter().enumerate().for_each(|(i, gamma)| {

            if let Some(R1) = self.R1.as_ref() {
                let R1NTTmut = R1NTT.as_mut().unwrap();
                ring.add_assign(&mut h,
                    mes.s1().as_ref().unwrap().iter().enumerate().fold(ring.zero(), |acc, (j, el)| {
                        let mut tmp = R::from_array(core::array::from_fn(|k|
                            ring.to_NTTRing_ref(R1.get(i, j*N + k))));
                        ring.ntt(&mut tmp);
                        ring.scalar_mul_assign_ref(&mut tmp, &gamma);
                        let res = ring.add(acc, ring.mul_ref(el, &tmp));
                        if i == 0 { R1NTTmut.push(tmp) }
                            else { ring.add_assign(&mut R1NTTmut[j], tmp) }
                        res
                 }));
            }

            // TODO: remove redundancy here!
            if let Some(Rm) = self.Rm.as_ref() {
                let RmNTTmut = RmNTT.as_mut().unwrap();
                ring.add_assign(&mut h,
                    mes.m().as_ref().unwrap().iter().enumerate().fold(ring.zero(), |acc, (j, el)| {
                        let mut tmp = R::from_array(core::array::from_fn(|k|
                            ring.to_NTTRing_ref(Rm.get(i, j*N + k))));
                        ring.ntt(&mut tmp);
                        ring.scalar_mul_assign_ref(&mut tmp, &gamma);
                        let res = ring.add(acc, ring.mul_ref(el, &tmp));
                        if i == 0 { RmNTTmut.push(tmp) }
                            else { ring.add_assign(&mut RmNTTmut[j], tmp) }
                        res
                 }));
            }

            ring.sub_assign(&mut h, ring.scalar_mul(ring.from_constant(&self.u[i]), &gamma));
        });

        RmNTT.as_mut().map(|x| x.push(ring.from_constant(&ring.base_ring().one())));

        let R1NTT = R1NTT.map(|x| DenseMatrixMul::new(ring, x.len(), x, "R1NTT"));
        let RmNTT = RmNTT.map(|x| DenseMatrixMul::new(ring, x.len(), x, "RmNTT"));

        (h, R1NTT, RmNTT)
    }

    // TODO: remove redundancy here!
    fn to_NTTform<'a>(&self, ring: &'a R, h: &El<R>, gammas: &[El<R::BaseRing>])
        -> (Option<DenseMatrixMul<'a, R>>, Option<DenseMatrixMul<'a, R>>, El<R>)
    {
        assert!(self.rows() == gammas.len());

        let mut R1NTT = self.R1.as_ref().map(|x| Vec::<El<R>>::with_capacity(x.columns()));
        let mut RmNTT = self.Rm.as_ref().map(|x| Vec::<El<R>>::with_capacity(x.columns() + 1));
        let mut uNTT = ring.clone_el(h);

        gammas.iter().enumerate().for_each(|(i, gamma)| {

            if let Some(R1) = self.R1.as_ref() {
                let R1NTTmut = R1NTT.as_mut().unwrap();
                (0..(R1.columns()/N)).for_each(|j| {
                    let mut tmp = R::from_array(core::array::from_fn(|k|
                        ring.to_NTTRing_ref(R1.get(i, j*N + k))));
                    ring.ntt(&mut tmp);
                    ring.scalar_mul_assign_ref(&mut tmp, &gamma);
                    if i == 0 { R1NTTmut.push(tmp) }
                        else { ring.add_assign(&mut R1NTTmut[j], tmp) }
                 });
            }

            // TODO: remove redundancy here!
            if let Some(Rm) = self.Rm.as_ref() {
                let RmNTTmut = RmNTT.as_mut().unwrap();
                (0..(Rm.columns()/N)).for_each(|j| {
                    let mut tmp = R::from_array(core::array::from_fn(|k|
                        ring.to_NTTRing_ref(Rm.get(i, j*N + k))));
                    ring.ntt(&mut tmp);
                    ring.scalar_mul_assign_ref(&mut tmp, &gamma);
                    if i == 0 { RmNTTmut.push(tmp) }
                        else { ring.add_assign(&mut RmNTTmut[j], tmp) }
                 });
            }

            ring.add_assign(&mut uNTT, ring.scalar_mul(ring.from_constant(&self.u[i]), &gamma));
        });

        RmNTT.as_mut().map(|x| x.push(ring.from_constant(&ring.base_ring().one())));

        let R1NTT = R1NTT.map(|x| DenseMatrixMul::new(ring, x.len(), x, "R1NTT"));
        let RmNTT = RmNTT.map(|x| DenseMatrixMul::new(ring, x.len(), x, "RmNTT"));

        (R1NTT, RmNTT, uNTT)
    }
}


struct LatSigmaPrecomp<R>
    where R: RingStore
{
    // NOTE: stored in NTT form
    y1: Option<Vec<El<R>>>,
    y2: Vec<El<R>>,
    w: Vec<El<R>>,
    By2: Option<Vec<El<R>>>
}

impl<R: RingStore> LatSigmaPrecomp<R> {
    fn empty() -> Self {
        Self { y1: None, y2: Vec::new(), w: Vec::new(), By2: None }
    }

    fn is_some(&self) -> bool {
        self.y1.is_some() || self.By2.is_some() || self.y2.len() > 0
    }

    fn into_inner(self) -> (Option<Vec<El<R>>>, Vec<El<R>>, Vec<El<R>>, Option<Vec<El<R>>>)
    {
        (self.y1, self.y2, self.w, self.By2)
    }
}


pub struct LatSigmaProof<R, const N: usize>
    where R: ABDLOPRingTrait<N>
{
    gammas: Vec<El<R::BaseRing>>,
    h: El<R>, // in NTT form
    z1: Option<Vec<El<R>>>, // in coeff form!
    z2: Vec<El<R>>, // in coeff form!
    w: Vec<El<R>>, // in NTT form
    vneg: Option<El<R>>, // in NTT form
    fscnt: usize
}


pub type LatSigmaDefault<'a, R, const N: usize> = LatSigma<'a, R,
        DenseMatrixMul<'a, <R as ABDLOPRingTrait<N>>::BaseRing>,
        SparseMatrixMul<'a, <R as ABDLOPRingTrait<N>>::BaseRing>, N>;

pub struct LatSigma<'a, R, MM1, MMm, const N: usize>
    where R: ABDLOPRingTrait<N>, MM1: MatrixMul<R = R::BaseRing>, MMm: MatrixMul<R = R::BaseRing>
{
    cs: ABDLOP<'a, R, N>,
    fs: RefCell<FiatShamirSim<FSRng>>,
    gamma: (Option<f64>, f64),
    challbnd: El<BigIntRing>, // TODO: add more general distributions besides inf bounds
    rsmode: RejSamplModes,
    linrel: RefCell<LatSigmaLinRel<R, MM1, MMm, N>>,
    precomp: RefCell<LatSigmaPrecomp<R>>
}

impl<'a, R, MM1, MMm, const N: usize> LatSigma<'a, R, MM1, MMm, N>
    where R: ABDLOPRingTrait<N>, MM1: MatrixMul<R = R::BaseRing>, MMm: MatrixMul<R = R::BaseRing>
{
    pub fn ring(&self) -> &R { self.cs.ring() }

    pub fn challbnd(&self) -> &El<BigIntRing> { &self.challbnd }

    pub fn abdlop(&self) -> &ABDLOP<'a, R, N> { &self.cs }

    pub fn get_fs(&self) -> &RefCell<FiatShamirSim<FSRng>> { &self.fs }

    fn get_sigma(&self, part: ABDLOPparts) -> f64 {
        ZZbig.to_float_approx(&self.challbnd) * (match part {
            ABDLOPparts::Ajtai =>
                self.gamma.0.unwrap() * ZZbig.to_float_approx(self.cs.get_bnd1().as_ref().unwrap()),
            ABDLOPparts::BDLOP =>
                self.gamma.1 * ZZbig.to_float_approx(&self.cs.get_bnd2())
        })
    }

    fn get_zbound(&self, part: ABDLOPparts) -> f64 {
        self.get_sigma(part) * (match part {
            ABDLOPparts::Ajtai => self.cs.get_m1().unwrap(),
            ABDLOPparts::BDLOP => self.cs.get_m2()
        } as f64).sqrt() * std::f64::consts::SQRT_2
    }

    pub fn set_linrel(&self, R1: Option<MM1>, Rm: Option<MMm>, u: Vec<El<R::BaseRing>>) {
        assert!(R1.is_some() || Rm.is_some());
        assert!(R1.is_none() || self.cs.has_ajtai());
        assert!(R1.as_ref().is_none_or(|x| x.columns() == self.cs.get_A1().unwrap().columns()*N));
        assert!(Rm.is_none() || self.cs.has_bdlop());
        assert!(Rm.as_ref().is_none_or(|x| x.columns() == self.cs.get_B().unwrap().rows()*N));
        assert!(R1.as_ref().is_none_or(|x| Rm.as_ref().is_none_or(|xx| x.rows() == xx.rows())));

        let mut linrelmut = self.linrel.borrow_mut();
        linrelmut.R1 = R1;
        linrelmut.Rm = Rm;
        linrelmut.u = u;
    }

    pub fn new(cs: ABDLOP<'a, R, N>,
        gamma: (Option<f64>, f64), challbnd: El<BigIntRing>, 
        rsmode: RejSamplModes
    ) -> Self {
        assert!(!cs.has_ajtai() || gamma.0.is_some());
        let fs = RefCell::new(FiatShamirSim::<FSRng>::new());
        Self { cs, fs, gamma, challbnd, rsmode,
            linrel: RefCell::new(LatSigmaLinRel::empty()),
            precomp: RefCell::new(LatSigmaPrecomp::empty()),
        }
    }

    pub fn prove(&self, com: &mut ABDLOPcommitment<R,N>, op: &ABDLOPopening<R,N>,
        mes: &ABDLOPmessage<R,N>) -> LatSigmaProof<R, N>
    {
        assert!(!self.cs.has_ajtai() || mes.s1().is_some());

        if !self.precomp.borrow().is_some() {
            println!("LatSigma: call precomp first for faster verification!");
            self.precomp()
        }

        let fsclone = self.fs.borrow().clone();
        let mut rng = self.cs.rng().borrow_mut();

        let mut g = self.ring().random_element(|| rng.next_u64());
        let gcoeff = R::to_array_mut(&mut g);
        gcoeff[0] = self.ring().NTT_ring().zero();
        self.cs.append_commit(com, op, &[self.ring().clone_el(&g)]);

        let linrel = self.linrel.borrow();
        let gammas = {
            let mut fsmut = self.fs.borrow_mut();
            gen_random(self.ring().base_ring(), fsmut.get_rng(), linrel.rows())
        };

        let (h, R1NTT, RmNTT) = linrel.compute_h(self.ring(), &g, &gammas, mes);

        let (mut y1, mut y2, w, By2) = self.precomp.replace(LatSigmaPrecomp::empty()).into_inner();

        let vneg = linrel.is_some().then(|| {
            let tmpv = RmNTT.map(|x| {
                let By2unwr = By2.as_ref().unwrap();
                x.mul(By2unwr).pop().unwrap()
            });
            if let Some(R1NTTunwr) = R1NTT {
                let y1unwr = y1.as_ref().unwrap();
                let R1y1 = R1NTTunwr.mul(y1unwr).pop().unwrap();
                tmpv.map_or(
                    self.ring().negate(self.ring().clone_el(&R1y1)),
                    |RmBy2| self.ring().sub_ref_snd(RmBy2, &R1y1)
                )
            } else {
                tmpv.unwrap()
            }
        });

        let tmpop = op.iter().map(|el| {
            let mut tmp = self.ring().clone_el(el);
            self.ring().intt(&mut tmp);
            tmp
        });
        let flatop = self.ring().to_base_ring(tmpop);
        self.ring().intt_vec(&mut y2);
        let flaty2 = self.ring().to_base_ring(y2.into_iter());

        let (z1, z2, fscnt) = if self.cs.has_ajtai() {
            let mut fsmut = self.fs.borrow_mut();

            self.ring().intt_vec(y1.as_mut().unwrap());
            let flaty1 = self.ring().to_base_ring(y1.unwrap().into_iter());

            let tmps1 = mes.s1().as_ref().unwrap().iter().map(|el| {
                let mut tmp = self.ring().clone_el(el);
                self.ring().intt(&mut tmp);
                tmp
            });
            let flats1 = self.ring().to_base_ring(tmps1.into_iter());

            let (zt, fscnt) = gen_vector_latrejsampl(self.ring().base_ring(), &mut rng, &mut fsmut,
                &self.challbnd, [self.gamma.0.unwrap(), self.gamma.1],
                [self.get_sigma(ABDLOPparts::Ajtai), self.get_sigma(ABDLOPparts::BDLOP)],
                self.rsmode, [&flaty1, &flaty2], [&flats1, &flatop]);
            let (z1, z2) = zt.into();
            (Some(self.ring().to_ntt_ring(z1.into_iter())), z2, fscnt)
        } else {
            let mut fsmut = self.fs.borrow_mut();
            let (zt, fscnt) = gen_vector_latrejsampl(self.ring().base_ring(), &mut rng, &mut fsmut,
                &self.challbnd, [self.gamma.1], [self.get_sigma(ABDLOPparts::BDLOP)],
                self.rsmode, [&flaty2], [&flatop]);
            let (z2,) = zt.into();
            (None, z2, fscnt)
        };

        self.fs.replace(fsclone);

        LatSigmaProof{ gammas, h, z1, z2: self.ring().to_ntt_ring(z2.into_iter()), w, vneg, fscnt }
    }

    pub fn precomp(&self) {
        let mut rng = self.cs.rng().borrow_mut();

        let mut y2 = self.ring().to_ntt_ring(gen_vector_dgauss(self.ring().base_ring(), &mut rng,
            self.get_sigma(ABDLOPparts::BDLOP), self.cs.get_A2().columns()*N).into_iter());
        self.ring().ntt_vec(&mut y2);

        let By2 = self.cs.get_B().as_ref().map(|B| B.mul(&y2));
        let (w, y1) = {
            let tmpw = self.cs.get_A2().mulit(&y2);
            if self.cs.has_ajtai() {
                let mut y1 = self.ring().to_ntt_ring(
                    gen_vector_dgauss(self.ring().base_ring(), &mut rng,
                        self.get_sigma(ABDLOPparts::Ajtai),
                        self.cs.get_A1().unwrap().columns()*N
                    ).into_iter());
                self.ring().ntt_vec(&mut y1);
                let resw = tmpw.zip(self.cs.get_A1().unwrap().mulit(&y1)).map(|(l, r)|
                    self.ring().add(l, r)).collect_vec();
                (resw, Some(y1))
            } else {
                (tmpw.collect_vec(), None)
            }
        };
        self.precomp.replace(LatSigmaPrecomp { y1, y2, w, By2 });
    }

    pub fn verify(&self, com: &ABDLOPcommitment<R,N>, proof: &LatSigmaProof<R,N>) -> bool {
        if !(proof.z1.is_some() == self.cs.has_ajtai() && com.len() == self.cs.comlen())
            { return false };
        
        let ring = self.ring();
        let basering = ring.base_ring();
        let intring = basering.integer_ring();

        let flatz1 = proof.z1.as_ref().map(|z1| self.ring().to_base_ring_ref(z1));
        if let Some(z1) = flatz1.as_ref() {
            if norm2(basering, &intring, &z1) > self.get_zbound(ABDLOPparts::Ajtai) { return false }
        }

        let flatz2 = self.ring().to_base_ring_ref(&proof.z2);
        if norm2(basering, &intring, &flatz2) > self.get_zbound(ABDLOPparts::BDLOP) { return false }
        
        let mut fs = self.fs.borrow_mut();
        let Rbnd = Zn::new(ZZbig, ZZbig.clone_el(&self.challbnd));
        let hom = basering.can_hom(&ZZbig).unwrap();
        let mut chall = basering.zero();
        (0..proof.fscnt).for_each(|_| chall = gen_infbnd(fs.get_rng(), &Rbnd, &hom));
        let chall = ring.from_constant(&chall);

        let mut z2 = self.ring().to_ntt_ring(flatz2.into_iter());
        self.ring().ntt_vec(&mut z2);
        let A2z2iter = self.cs.get_A2().mulit(&z2);

        let z1 = flatz1.map(|x| {
            let mut tmp = self.ring().to_ntt_ring(x.into_iter());
            self.ring().ntt_vec(&mut tmp);
            tmp
        });
        let lhsiter = if let Some(z1) = z1.as_ref() {
            Box::new(A2z2iter.zip(self.cs.get_A1().unwrap().mulit(z1)).map(|(A2z2i, A1z1i)|
                ring.add(A2z2i, A1z1i))) as Box<dyn Iterator<Item = El<R>>>
        } else { Box::new(A2z2iter) };

        let m1 = self.cs.get_A2().rows();
        if izip!(lhsiter, &proof.w, &com[..m1]).any(|(lhsi, wi, ci)|
            !ring.eq_el(&lhsi, &ring.add_ref_fst(wi, ring.mul_ref(&chall, ci))))
        { return false }

        let linrel = self.linrel.borrow();
        if linrel.is_some() {

            let (R1NTT, RmNTT, uNTT) = linrel.to_NTTform(self.ring(), &proof.h, &proof.gammas);

            let challu = ring.mul_ref_snd(uNTT, &chall);
            let lhs = if let Some(R1) = R1NTT.as_ref() {
                ring.add(challu, R1.mul(z1.as_ref().unwrap()).pop().unwrap())
            } else { challu };

            let lhs2 = if let Some(Rm) = RmNTT.as_ref() { 
                let Bz2iter = self.cs.get_B().unwrap().mulit(&z2);
                let RmBz2 = Bz2iter.zip(Rm.data()).fold(ring.zero(), |acc, (l, r)|
                    ring.add(acc, ring.mul_ref_snd(l, r)));
                let Rmt = Rm.mul(&com[m1..]).pop().unwrap();
                ring.sub(ring.add(lhs, RmBz2), ring.mul(Rmt, chall))
            } else { lhs };

            if !ring.eq_el(&lhs2, proof.vneg.as_ref().unwrap()) { return false }
        }
        return true
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use feanor_math::ring::RingValue;
    use feanor_math::homomorphism::Homomorphism;

    use crate::{
        lattice::gen_vector_infbnd,
        commit::abdlop::{ABDLOPRing, ABDLOPRingBase, ABDLOPmessage},
    };


    #[test]
    fn test_latsigma() {

        // use feanor_math::integer::IntegerRing;
        // let p = ZZbig.get_ring().parse("864175120484581453683482079962486176185193500155369104423588921177379322250834082489183304374038697487834084609675858746433355728113743766078731283595263", 10).unwrap();
        // let ring = feanor_math::rings::zn::zn_big::Zn::new(ZZbig, p);
        // let field = ring.clone().as_field().ok().unwrap();
        // type FieldImpl = AsField<feanor_math::rings::zn::zn_big::Zn<BigIntRing>>;

        let ring = feanor_math::rings::zn::zn_64::Zn::new(65537);

        const N: usize = 1 << 12;
        let abdlopring = RingValue::from(ABDLOPRingBase::<_, N>::new_promise_is_perfect_field(ring.clone()));
        
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
        
        let rng = <rand::rngs::StdRng as rand::SeedableRng>::from_os_rng();
        let abdlop = ABDLOP::random(&abdlopring, rng, n, Some(l), m1, m2, bnd1, bnd2);

        // let mut rng = rand::rng();
        let mut rng = <rand::rngs::StdRng as rand::SeedableRng>::from_os_rng();
        let s1 = abdlop.gen_s1(gen_vector_infbnd(&ring, &mut rng,
                abdlop.get_bnd1().as_ref().unwrap(), m2*N));
        let m = abdlop.gen_m(gen_random(&ring, &mut rng, l*N));
        let mes = ABDLOPmessage::new(&abdlopring, Some(s1), Some(m));

        let (mut com, op) = abdlop.commit(&mes);

        assert!(abdlop.open(&com, &mes, &op));

        // NOTE: we set gamma here to ensure M\approx 3
        let rsmode = RejSamplModes::Mode1;
        let gamma1 = Some(13f64);
        let gamma2 = 13f64;
        // let rsmode = RejSamplModes::Mode2;
        // let gamma1 = Some(0.675f64);
        // let gamma2 = 0.675f64;
        let gamma = (gamma1, gamma2);
        let challbnd = ZZbig.power_of_two(128);

        let latsigma: LatSigmaDefault<ABDLOPRing<_, N>, N>
            = LatSigma::new(abdlop, gamma, challbnd, rsmode);

        let proof = latsigma.prove(&mut com, &op, &mes);

        assert!(latsigma.verify(&com, &proof));
    }
}
