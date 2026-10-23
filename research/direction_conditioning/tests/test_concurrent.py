import sys,threading,time,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from concurrent_server import dispatch

class Concurrent(unittest.TestCase):
    def test_fifty_parallel_once(self):
        lock=threading.Lock();barrier=threading.Barrier(50,timeout=10)
        active=0;peak=0;seen=[]
        def worker(job,deadline):
            nonlocal active,peak
            with lock:
                active+=1;peak=max(peak,active);seen.append(job['id'])
            if job['id']<50:barrier.wait()
            time.sleep(.005)
            with lock:active-=1
            return {'status':'valid'}
        result=dispatch([{'id':i} for i in range(60)],worker,50,60,20)
        self.assertEqual(result['status'],'complete')
        self.assertEqual(peak,50)
        self.assertEqual(sorted(seen),list(range(60)))

    def test_fail_stops_new_requests(self):
        result=dispatch([{'id':i} for i in range(5)],lambda j,d:{'status':'failed'},1,5,10)
        self.assertEqual(result['admitted'],[0])
        self.assertEqual(result['unsent'],[1,2,3,4])

    def test_budget_before_call(self):
        def forbidden(*args):raise AssertionError('must not call')
        with self.assertRaises(ValueError):dispatch([{'id':1},{'id':2}],forbidden,2,1,10)
        with self.assertRaises(ValueError):dispatch([{'id':1},{'id':1}],forbidden,2,2,10)
        with self.assertRaises(ValueError):dispatch([],forbidden,51,1,10)

if __name__=='__main__':unittest.main()
