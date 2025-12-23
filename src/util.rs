use rand::RngCore;
use rand_seeder::{Seeder, SipRng};

use feanor_math::ring::{El, RingStore, RingExtension};
use feanor_math::rings::finite::{FiniteRingStore, FiniteRing};

pub type CoeffRing<P> = <<P as RingStore>::Type as RingExtension>::BaseRing;
pub type Coeff<P> = El<CoeffRing<P>>;


pub fn gen_vector<EL>(mut f: impl FnMut() -> EL, len: usize)
    -> Vec<EL> {
    (0..len).map(|_| f()).collect()
}


// TODO: check where we can use this instead of calling bits_from_int for all ints
pub fn bits(bitlen: usize) -> impl Iterator<Item = Vec<usize>>
{
    let mut cur = vec![1; bitlen];
    let mut cnt = 0;
    std::iter::from_fn(move || {
        let mut carry = 1;
        (0..bitlen).for_each(|i| {
            let tmp = cur[i] ^ carry;
            carry &= cur[i];
            cur[i] = tmp;
        });
        cnt += 1;
        if cnt > 1 << bitlen { None }
        else { Some(cur.clone()) }
    })
}

// outputs bits from lsb to msb
pub fn bits_from_int(inp: usize, bitlen: usize)
    -> impl ExactSizeIterator<Item = usize> + DoubleEndedIterator<Item = usize> + Clone
{
    (0..bitlen).map(move |n| (inp >> n) & 1)
}


// outputs integer from lsb to msb bits
pub fn int_from_bits<I>(bits: I) -> usize
    where I: Iterator<Item = usize>
{
    bits.enumerate().fold(0, |acc, (i, b)| if b != 0 {acc + (1 << i)} else {acc})
}


pub struct FiatShamirSim<'a, R> {
    ring: &'a R,
    rng: SipRng
}

impl<'a, R: RingStore<Type: FiniteRing>> FiatShamirSim<'a, R> {

    fn get_rng() -> SipRng {
        Seeder::from("FiatShamirSim").into_rng()
    }

    pub fn new(ring: &'a R) -> Self {
        Self {
            ring,
            rng: Self::get_rng()
        }
    }

    pub fn challenge(&mut self) -> El<R> {
        let mut el = self.ring.random_element(|| self.rng.next_u64());
        while self.ring.is_zero(&el) {
            el = self.ring.random_element(|| self.rng.next_u64());
        }
        // println!("FiatShamirSim new chall: {}", self.ring.format(&el));
        el
    }

    pub fn reset(&mut self) {
        self.rng = Self::get_rng()
    }

}

// TODO: why is this not possible with derive?
impl<'a, R: RingStore> Clone for FiatShamirSim<'a, R> {
    fn clone(&self) -> Self {
        Self {
            ring: self.ring,
            rng: self.rng.clone()
        }
    } 
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use rand::Rng;
    use feanor_math::rings::zn::ZnRingStore;
    use feanor_math::rings::zn::zn_64::Zn;

    pub fn gen_random<R>(ring: &R, len: usize, seed: Option<&str>) -> Vec<El<R>>
        where R: RingStore<Type: FiniteRing>
    {
        if let Some(seed) = seed {
            let mut rng: SipRng = Seeder::from(seed).into_rng();
            gen_vector::<El<R>>(|| ring.random_element(|| rng.next_u64()), len)
        } else {
            gen_vector::<El<R>>(|| ring.random_element(rand::random::<u64>), len)
        }
    }

    pub fn test_rot<R: RingStore>(ring: &R, inp: &Vec<El<R>>, out: &Vec<El<R>>, by: usize) {
        assert!(inp.len() == out.len());
        assert!((0..inp.len()).all(|i| ring.eq_el(&inp[i], &out[(i+by)%out.len()])))
    }

    #[test]
    fn test_bits() {

        let N = 7;
        let int = rand::rng().random_range(0..(1 << N));
        
        assert_eq!(int_from_bits((0..N).map(|_| 0)), 0);
        assert_eq!(int_from_bits((0..N).map(|_| 1)), (1 << N) - 1);
        assert_eq!(int_from_bits(bits_from_int(int, N)), int);
        assert!(bits(N).count() == 1 << N);
        assert!(bits(N).zip(0..(1 << N)).all(|(bv, i)|
            bits_from_int(i, N).enumerate().all(|(ind, b)| bv[ind] == b)));
    }

    #[test]
    fn test_fiatshamirsim() {

        let N = 10;
        let field = Zn::new(65537).as_field().ok().unwrap();

        let mut fs1 = FiatShamirSim::new(&field);
        let mut fs2 = FiatShamirSim::new(&field);

        (0..N).all(|_| field.eq_el(&fs1.challenge(), &fs2.challenge()));

        (0..N).for_each(|_| {fs1.challenge();});

        let mut fs3 = fs1.clone();

        (0..N).all(|_| field.eq_el(&fs1.challenge(), &fs3.challenge()));
    }
}
